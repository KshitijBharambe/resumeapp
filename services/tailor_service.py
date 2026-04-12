import base64
import json
import os

import requests
from flask import jsonify

from config import (
    ANTHROPIC_MODELS,
    DEFAULT_RESUME,
    OUTPUT_FOLDER,
    PROVIDER_BASE_URLS,
    PROVIDER_DISPLAY,
    SYSTEM_PROMPT,
)
from services.resume_service import (
    apply_replacements,
    build_tailor_message,
    enrich_with_sections,
    extract_json_array,
    extract_resume_text,
    extract_role_suggestion,
    filter_replacements_by_type,
    get_resume_paragraphs,
    make_output_name,
    name_prefix,
    normalize_replacement_keys,
)

JOB_DESCRIPTION_EMPTY_ERROR = "Job description cannot be empty."
NO_RESUME_ERROR = "No resume found - please upload one first."
JSON_CONTENT_TYPE = "application/json"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _load_resume_context():
    try:
        resume_text = extract_resume_text(DEFAULT_RESUME)
        resume_paras = get_resume_paragraphs(DEFAULT_RESUME)
    except Exception as pre_err:
        return None, None, jsonify({"error": f"Failed to read resume: {pre_err}"}), 500
    return resume_text, resume_paras, None, None


def _parse_tailor_response(raw, resume_paras, repair_callback=None):
    raw = strip_think(raw)
    replacements = extract_json_array(raw)
    role_suggestions = None

    if replacements is not None:
        role_suggestions, replacements = extract_role_suggestion(replacements)
        replacements = normalize_replacement_keys(replacements)
        replacements = filter_replacements_by_type(replacements, resume_paras)

    if not replacements and repair_callback is not None:
        repair_raw = repair_callback(raw)
        if repair_raw:
            repair_raw = strip_think(repair_raw)
            print("[DEBUG] Repair output (first 500):", repair_raw[:500])
            replacements = extract_json_array(repair_raw)
            if replacements is not None:
                _role_suggestions, replacements = extract_role_suggestion(replacements)
                if _role_suggestions and not role_suggestions:
                    role_suggestions = _role_suggestions
                replacements = normalize_replacement_keys(replacements)
                replacements = filter_replacements_by_type(replacements, resume_paras)

    return raw, role_suggestions, replacements


def _finalize_tailored_output(replacements, role_suggestions, resume_paras, job_title):
    enriched = enrich_with_sections(replacements, resume_paras)
    try:
        prefix = name_prefix(resume_paras)
        output_name = make_output_name(job_title, prefix=prefix)
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        apply_replacements(DEFAULT_RESUME, output_path, replacements)
    except Exception as error:
        return jsonify({"error": f"Failed to write .docx: {error}"}), 500

    return jsonify(
        {
            "success": True,
            "filename": output_name,
            "docx_b64": read_docx_b64(output_path),
            "changes_count": len(enriched),
            "changes": enriched,
            "role_suggestions": role_suggestions,
        }
    )


def _collect_streamed_raw(response, initial_raw=""):  # NOSONAR
    raw = initial_raw
    finish_reason = None
    stream_error = None
    for line in response.iter_lines():  # NOSONAR
        if not line:
            continue
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if line.startswith("event:"):
            continue
        if line.startswith("data: "):
            line = line[6:]
        if line.startswith(":"):
            continue
        if line.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(line)
        except Exception as error:
            print(f"[WARN] Failed to parse stream chunk: {line} - Error: {error}")
            continue
        if "error" in chunk and "choices" not in chunk:
            err_msg = chunk["error"]
            if isinstance(err_msg, dict):
                err_msg = err_msg.get("message", str(err_msg))
            stream_error = str(err_msg)
            print(f"[ERROR] Stream returned error: {stream_error}")
            break
        choice = chunk["choices"][0]
        raw += choice["delta"].get("content", "")
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
    return raw, finish_reason, stream_error


def _stream_error_response(stream_error):
    if (
        "context length" in stream_error.lower()
        or "tokens to keep" in stream_error.lower()
    ):
        return (
            jsonify(
                {
                    "error": "Input too large for this model's context window.",
                    "hint": (
                        "The resume + job description + system prompt exceeds what the model can handle. "
                        "In LM Studio, go to the model's settings and increase the context length "
                        "(32,768 or higher recommended). The Context Length setting in this app "
                        "only works if the model is loaded with enough context in LM Studio."
                    ),
                }
            ),
            400,
        )
    return jsonify({"error": f"Model backend error: {stream_error}"}), 500


def get_chat_url_and_headers(provider, lm_url, api_key):
    """Return (chat_completions_url, extra_headers) for the given provider."""
    local_providers = {"lmstudio", "ollama", "custom"}
    if provider in local_providers:
        base = (
            lm_url or PROVIDER_BASE_URLS.get(provider, "http://localhost:1234")
        ).rstrip("/")
    else:
        base = PROVIDER_BASE_URLS.get(provider, "").rstrip("/")

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://resume-tailor.onrender.com"
    return f"{base}/v1/chat/completions", headers


def _provider_request_values(data):
    provider = data.get("provider", "lmstudio").lower().strip()
    api_key = data.get("api_key", "").strip()
    lm_url = data.get("lm_url", "").strip()
    if provider == "lmstudio" and not api_key:
        api_key = os.environ.get("LM_API_TOKEN", "")
    return provider, api_key, lm_url


def _platform_from_compatibility(compatibility):
    if compatibility == "mlx":
        return "mac"
    if compatibility in ("gguf", "exl2"):
        return "desktop"
    return ""


def _list_gemini_models(api_key):
    if not api_key:
        return jsonify({"error": "API key required"}), 400
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        raw = list(client.models.list())
        models = []
        for model in raw:
            model_name = str(getattr(model, "name", "") or "")
            name = model_name.split("/")[-1] if "/" in model_name else model_name
            if not any(token in name for token in ["gemini", "learnlm"]):
                continue
            if "embedding" in name or "vision" in name:
                continue
            token_limit = getattr(model, "input_token_limit", None) or getattr(
                model, "inputTokenLimit", None
            )
            models.append({"id": name, "input_token_limit": token_limit})
        models.sort(key=lambda item: item["id"])
        return jsonify({"models": models})
    except Exception as error:
        return jsonify({"error": str(error)}), 500


def _list_local_provider_models(provider, lm_url, api_key):
    base = (lm_url or PROVIDER_BASE_URLS.get(provider, "http://localhost:1234")).rstrip(
        "/"
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    if provider == "ollama":
        try:
            response = requests.get(f"{base}/api/tags", headers=headers, timeout=4)
            response.raise_for_status()
            raw = response.json().get("models", [])
            models = [{"id": model["name"]} for model in raw]
            return jsonify({"models": models, "online": True})
        except Exception:
            pass

    try:
        response = requests.get(f"{base}/v1/models", headers=headers, timeout=4)
        response.raise_for_status()
        raw = response.json().get("data", [])
        models = []
        for model in raw:
            compatibility = model.get("compatibility_type", "")
            platform = _platform_from_compatibility(compatibility)
            models.append(
                {
                    "id": model["id"],
                    "platform": platform,
                    "compat": compatibility,
                    "quant": model.get("quantization", ""),
                }
            )
        return jsonify({"models": models, "online": True})
    except Exception:
        return jsonify({"models": [], "online": False})


def _list_cloud_provider_models(provider, api_key):
    base = PROVIDER_BASE_URLS.get(provider, "").rstrip("/")
    if not base:
        return jsonify({"error": f"Unknown provider: {provider}"}), 400
    if not api_key:
        return jsonify({"error": "API key required"}), 400

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{base}/v1/models", headers=headers, timeout=8)
        if response.status_code == 401:
            return jsonify({"error": "Invalid API key"}), 401
        response.raise_for_status()
        raw = response.json().get("data", [])
        chat_keywords = [
            "gpt",
            "claude",
            "llama",
            "mistral",
            "mixtral",
            "gemma",
            "qwen",
            "deepseek",
            "command",
            "sonar",
            "hermes",
            "nous",
        ]
        models = []
        for model in raw:
            model_id = model.get("id", "")
            is_chat_model = any(
                keyword in model_id.lower() for keyword in chat_keywords
            )
            if is_chat_model or provider in ("groq", "mistral"):
                models.append({"id": model_id})
        if not models:
            models = [{"id": model.get("id", "")} for model in raw]
        models.sort(key=lambda item: item["id"])
        return jsonify({"models": models})
    except requests.exceptions.ConnectionError:
        return (
            jsonify(
                {
                    "error": f"Cannot connect to {PROVIDER_DISPLAY.get(provider, provider)}"
                }
            ),
            503,
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 500


def provider_models(data):
    """Unified model listing for all providers."""
    provider, api_key, lm_url = _provider_request_values(data)

    if provider == "anthropic":
        return jsonify({"models": ANTHROPIC_MODELS, "static": True})
    if provider == "gemini":
        return _list_gemini_models(api_key)
    if provider in ("lmstudio", "ollama", "custom"):
        return _list_local_provider_models(provider, lm_url, api_key)
    return _list_cloud_provider_models(provider, api_key)


def tailor_resume(data):
    provider = data.get("provider", "lmstudio").lower().strip()
    if provider == "gemini":
        return _tailor_gemini_impl(data)
    if provider == "anthropic":
        return _tailor_anthropic(data)
    return _tailor_openai_compatible(data)


def _tailor_anthropic(data):
    """Handle resume tailoring via the Anthropic Messages API."""
    api_key = data.get("api_key", "").strip()
    model = data.get("model", "claude-sonnet-4-6").strip()
    jd_text = data.get("jd_text", "").strip()
    job_title = data.get("job_title", "").strip()
    max_tokens = int(data.get("max_tokens", 8192))

    if not api_key:
        return jsonify({"error": "Anthropic API key is required."}), 400
    if not jd_text:
        return jsonify({"error": JOB_DESCRIPTION_EMPTY_ERROR}), 400
    if not os.path.exists(DEFAULT_RESUME):
        return jsonify({"error": NO_RESUME_ERROR}), 400

    resume_text, resume_paras, error_response, error_code = _load_resume_context()
    if error_response:
        return error_response, error_code
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    combined_message = build_tailor_message(resume_text, resume_paras, jd_text)

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": combined_message}],
            },
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": JSON_CONTENT_TYPE,
            },
            timeout=300,
        )
        if response.status_code == 401:
            return (
                jsonify(
                    {"error": "Invalid Anthropic API key. Check console.anthropic.com."}
                ),
                401,
            )
        if response.status_code == 429:
            return (
                jsonify(
                    {"error": "Anthropic API rate limit exceeded. Try again shortly."}
                ),
                429,
            )
        response.raise_for_status()
        raw = response.json()["content"][0]["text"]
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Anthropic API."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Anthropic API timed out."}), 504
    except Exception as error:
        return jsonify({"error": f"Anthropic API error: {error}"}), 500

    raw, role_suggestions, replacements = _parse_tailor_response(raw, resume_paras)
    if replacements is None or not replacements:
        preview = raw[:1200] if raw else "(empty)"
        return (
            jsonify(
                {
                    "error": "Could not extract JSON array from Anthropic response.",
                    "hint": "Model returned prose without a JSON array.",
                    "raw_preview": preview,
                }
            ),
            500,
        )

    return _finalize_tailored_output(
        replacements, role_suggestions, resume_paras, job_title
    )


def _tailor_openai_compatible(data):  # NOSONAR
    provider = data.get("provider", "lmstudio").lower().strip()
    jd_text = data.get("jd_text", "").strip()
    model = data.get("model", "").strip()
    job_title = data.get("job_title", "").strip()
    api_key = data.get("api_key", "").strip()  # NOSONAR
    lm_url = data.get("lm_url", "").strip()

    if provider == "lmstudio" and not api_key:
        api_key = os.environ.get("LM_API_TOKEN", "")

    temperature = float(data.get("temperature", 0.3))
    max_tokens = int(data.get("max_tokens", 8192))
    top_p = float(data.get("top_p", 0.95))
    top_k = data.get("top_k")
    repeat_penalty = float(data.get("repeat_penalty", 1.1))
    seed = int(data.get("seed", -1))
    context_length = data.get("context_length")

    if not jd_text:
        return jsonify({"error": JOB_DESCRIPTION_EMPTY_ERROR}), 400
    if not os.path.exists(DEFAULT_RESUME):
        return jsonify({"error": NO_RESUME_ERROR}), 400

    chat_url, extra_headers = get_chat_url_and_headers(provider, lm_url, api_key)
    provider_label = PROVIDER_DISPLAY.get(provider, provider)

    # Quick sanity check: if this is a cloud provider like OpenRouter, confirm
    # the requested model exists on the backend to avoid confusing stream
    # parsing errors when the user accidentally supplies a model name that
    # isn't hosted by that provider.
    try:
        base = (lm_url or PROVIDER_BASE_URLS.get(provider, "")).rstrip("/")
        if provider in {"openrouter", "openai", "groq", "mistral"} and base:
            try:
                model_list_resp = requests.get(
                    f"{base}/v1/models", headers=extra_headers, timeout=6
                )
                if model_list_resp.ok:
                    data = model_list_resp.json()
                    raw_models = data.get("data") or data.get("models") or []
                    available_ids = [
                        m.get("id") or m.get("name")
                        for m in raw_models
                        if isinstance(m, dict)
                    ]
                    if model and model not in available_ids:
                        return (
                            jsonify(
                                {
                                    "error": f"Model '{model}' not found on {provider_label}.",
                                    "hint": "Confirm the model name or load available models via 'Load Models' before running.",
                                }
                            ),
                            404,
                        )
            except Exception:
                # If the models endpoint is unavailable or times out, continue —
                # we'll surface any backend errors later when attempting the chat call.
                pass
    except Exception:
        pass

    resume_text, resume_paras, error_response, error_code = _load_resume_context()
    if error_response:
        return error_response, error_code
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    combined_message = build_tailor_message(resume_text, resume_paras, jd_text)

    local_providers = {"lmstudio", "ollama", "custom"}
    if provider in local_providers and "/no_think" not in combined_message:
        combined_message = combined_message + "\n/no_think"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined_message},
    ]
    if provider in local_providers:
        messages.append({"role": "assistant", "content": '[{"original":'})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": True,
    }
    cloud_providers = {"openai", "groq", "openrouter", "mistral"}
    if provider in cloud_providers:
        payload["response_format"] = {"type": "json_object"}
    if provider in local_providers:
        payload["repeat_penalty"] = repeat_penalty
        if top_k and int(top_k) > 0:
            payload["top_k"] = int(top_k)
        if seed != -1:
            payload["seed"] = seed
        payload["context_length"] = int(context_length) if context_length else 32768

    response = None
    try:
        response = requests.post(
            chat_url, json=payload, headers=extra_headers, timeout=600, stream=True
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as error:
        print(f"[ERROR] ConnectionError connecting to {provider_label}: {error}")
        return (
            jsonify(
                {
                    "error": f"Cannot connect to {provider_label}. Check the URL is reachable."
                }
            ),
            503,
        )
    except requests.exceptions.Timeout as error:
        print(f"[ERROR] Timeout connecting to {provider_label}: {error}")
        return (
            jsonify(
                {
                    "error": f"{provider_label} timed out (10 min). Model may be too slow."
                }
            ),
            504,
        )
    except Exception as error:
        print(f"[ERROR] Exception connecting to {provider_label}: {error}")
        return jsonify({"error": f"{provider_label} error: {error}"}), 500

    finish_reason = None
    stream_error = None
    try:
        raw, finish_reason, stream_error = _collect_streamed_raw(
            response, initial_raw='[{"original":' if provider in local_providers else ""
        )
    except Exception as error:
        print(f"[ERROR] Stream reading failed: {error}")
        return jsonify({"error": f"Error reading stream: {error}"}), 500
    finally:
        if response:
            response.close()

    if stream_error:
        return _stream_error_response(stream_error)

    print(f"[DEBUG] finish_reason={finish_reason}, raw length={len(raw)} chars")

    stripped_raw = raw.strip().strip("[").strip()
    if not stripped_raw:
        hint = "The model returned an empty response."
        if finish_reason == "length":
            hint += (
                " Output was cut off by the token limit — try increasing Max Tokens."
            )
        else:
            hint += (
                " This usually means the input exceeded the model's context window."
                " Try a model with a larger context, or increase the Context Length setting."
            )
        return jsonify({"error": hint}), 500

    print("[DEBUG] Raw output (first 800):", raw[:800])

    def _repair_openai_response(current_raw):
        print("[INFO] First parse failed, attempting repair request...")
        repair_messages = [
            {
                "role": "system",
                "content": "You are a JSON repair assistant. Output ONLY a valid JSON array, nothing else.",
            },
            {
                "role": "user",
                "content": (
                    "The following text was supposed to be a JSON array of objects with "
                    '"original" and "replacement" keys, but it could not be parsed. '
                    "Extract the data and return ONLY a valid JSON array. "
                    "No markdown fences, no explanation.\n\n" + current_raw[:4000]
                ),
            },
        ]
        repair_payload = {
            "model": model,
            "messages": repair_messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if provider in cloud_providers:
            repair_payload["response_format"] = {"type": "json_object"}
        try:
            repair_response = requests.post(
                chat_url, json=repair_payload, headers=extra_headers, timeout=120
            )
            repair_response.raise_for_status()
            return repair_response.json()["choices"][0]["message"]["content"]
        except Exception as error:
            print(f"[WARN] Repair request failed: {error}")
            return None

    raw, role_suggestions, replacements = _parse_tailor_response(
        raw,
        resume_paras,
        repair_callback=_repair_openai_response,
    )

    if not replacements:
        preview = raw[:1200] if raw else "(empty — model produced no output)"
        if finish_reason == "length":
            error_message = (
                "Model output was truncated (hit token limit) — the JSON was cut off."
            )
            hint_message = "Increase Max Tokens, increase Context Length, or use a model with a larger context window."
        else:
            error_message = "Could not extract a JSON array from the model response."
            hint_message = "The model may have returned analysis/prose without a JSON array. Check the raw output below."
        return (
            jsonify(
                {"error": error_message, "hint": hint_message, "raw_preview": preview}
            ),
            500,
        )

    return _finalize_tailored_output(
        replacements, role_suggestions, resume_paras, job_title
    )


def _tailor_gemini_impl(data):  # NOSONAR
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return (
            jsonify(
                {
                    "error": "google-genai is not installed. Run: pip install google-genai"
                }
            ),
            500,
        )

    api_key = data.get("api_key", "").strip()
    model_name = data.get("model", "gemini-3.1-pro-preview").strip()
    jd_text = data.get("jd_text", "").strip()
    job_title = data.get("job_title", "").strip()
    temperature = float(data.get("temperature", 0.3))
    max_tokens = int(data.get("max_tokens", 8192))
    top_p = float(data.get("top_p", 0.95))

    if not api_key:
        return jsonify({"error": "Gemini API key is required."}), 400
    if not jd_text:
        return jsonify({"error": JOB_DESCRIPTION_EMPTY_ERROR}), 400
    if not os.path.exists(DEFAULT_RESUME):
        return jsonify({"error": NO_RESUME_ERROR}), 400

    resume_text, resume_paras, error_response, error_code = _load_resume_context()
    if error_response:
        return error_response, error_code
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    prompt = build_tailor_message(resume_text, resume_paras, jd_text)
    thinking_models = {"gemini-2.5-pro", "gemini-2.5-flash", "gemini-3"}

    try:
        client = genai.Client(api_key=api_key)
        use_thinking = any(model_name.startswith(prefix) for prefix in thinking_models)

        if use_thinking:
            config = genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=1,
                max_output_tokens=max_tokens,
                top_p=top_p,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=8192),
            )
        else:
            config = genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
                response_mime_type=JSON_CONTENT_TYPE,
            )

        response = client.models.generate_content(
            model=model_name, contents=prompt, config=config
        )
        raw = response.text or ""
    except Exception as error:
        err_str = str(error)
        if (
            "API_KEY_INVALID" in err_str
            or "api key" in err_str.lower()
            or "401" in err_str
        ):
            return (
                jsonify(
                    {
                        "error": "Invalid Gemini API key. Check your key at aistudio.google.com."
                    }
                ),
                401,
            )
        if "quota" in err_str.lower() or "429" in err_str:
            return (
                jsonify(
                    {
                        "error": "Gemini API quota exceeded. Try again later or use a different key."
                    }
                ),
                429,
            )
        if "not found" in err_str.lower() or "404" in err_str:
            return (
                jsonify(
                    {"error": f"Model '{model_name}' not found. Check the model name."}
                ),
                404,
            )
        return jsonify({"error": f"Gemini API error: {err_str}"}), 500

    print("[DEBUG][Gemini] Raw output (first 800):", raw[:800])

    def _repair_gemini_response(current_raw):
        print("[INFO][Gemini] First parse failed, attempting repair request...")
        try:
            repair_config = genai_types.GenerateContentConfig(
                system_instruction="You are a JSON repair assistant. Output ONLY a valid JSON array, nothing else.",
                temperature=0.0,
                max_output_tokens=max_tokens,
                response_mime_type=JSON_CONTENT_TYPE,
            )
            repair_prompt = (
                "The following text was supposed to be a JSON array of objects with "
                '"original" and "replacement" keys, but it could not be parsed. '
                "Extract the data and return ONLY a valid JSON array. "
                "No markdown fences, no explanation.\n\n" + current_raw[:4000]
            )
            repair_response = client.models.generate_content(
                model=model_name, contents=repair_prompt, config=repair_config
            )
            return repair_response.text
        except Exception as error:
            print(f"[WARN][Gemini] Repair request failed: {error}")
            return None

    raw, role_suggestions, replacements = _parse_tailor_response(
        raw,
        resume_paras,
        repair_callback=_repair_gemini_response,
    )

    if not replacements:
        preview = raw[:1200] if raw else "(empty)"
        return (
            jsonify(
                {
                    "error": "Could not extract a JSON array from the Gemini response.",
                    "hint": "The model may have returned prose without a JSON array.",
                    "raw_preview": preview,
                }
            ),
            500,
        )

    return _finalize_tailored_output(
        replacements, role_suggestions, resume_paras, job_title
    )


def strip_think(text):  # NOSONAR
    """
    Strip <think>...</think> blocks from Qwen3/DeepSeek output.
    Handles: complete blocks, orphaned </think>, truncated blocks.
    """
    if not text:
        return text
    if THINK_OPEN in text and THINK_CLOSE in text:
        cleaned = (
            __import__("re")
            .sub(r"<think>.*?</think>", "", text, flags=__import__("re").DOTALL)
            .strip()
        )
        if cleaned:
            return cleaned
    if THINK_CLOSE in text and THINK_OPEN not in text:
        after = text.split(THINK_CLOSE, 1)[1].strip()
        return after if after else text.split(THINK_CLOSE, 1)[0].strip()
    if THINK_OPEN in text:
        for marker in ["{", "["]:
            idx = text.rfind(marker)
            if idx != -1:
                candidate = text[idx:].strip()
                if (marker == "{" and "}" in candidate) or (
                    marker == "[" and "]" in candidate
                ):
                    return candidate
        return ""
    return text.strip()


def read_docx_b64(path):
    """Read a .docx file and return it base64-encoded (for localStorage caching)."""
    with open(path, "rb") as file_handle:
        return base64.b64encode(file_handle.read()).decode("utf-8")
