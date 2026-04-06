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


def provider_models(data):
    """Unified model listing for all providers."""
    provider = data.get("provider", "lmstudio").lower().strip()
    api_key = data.get("api_key", "").strip()
    lm_url = data.get("lm_url", "").strip()

    if provider == "lmstudio" and not api_key:
        api_key = os.environ.get("LM_API_TOKEN", "")

    if provider == "anthropic":
        return jsonify({"models": ANTHROPIC_MODELS, "static": True})

    if provider == "gemini":
        if not api_key:
            return jsonify({"error": "API key required"}), 400
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            raw = list(client.models.list())
            models = []
            for model in raw:
                name = model.name.split("/")[-1] if "/" in model.name else model.name
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

    if provider in ("lmstudio", "ollama", "custom"):
        base = (
            lm_url or PROVIDER_BASE_URLS.get(provider, "http://localhost:1234")
        ).rstrip("/")
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
                platform = (
                    "mac"
                    if compatibility == "mlx"
                    else ("desktop" if compatibility in ("gguf", "exl2") else "")
                )
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

    base = PROVIDER_BASE_URLS.get(provider, "").rstrip("/")
    if not base:
        return jsonify({"error": f"Unknown provider: {provider}"}), 400

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if not api_key:
        return jsonify({"error": "API key required"}), 400
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
            if any(
                keyword in model_id.lower() for keyword in chat_keywords
            ) or provider in ("groq", "mistral"):
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
        return jsonify({"error": "Job description cannot be empty."}), 400
    if not os.path.exists(DEFAULT_RESUME):
        return jsonify({"error": "No resume found - please upload one first."}), 400

    # Pre-validate: parse docx and ensure output folder is writable before spending API credits
    try:
        resume_text = extract_resume_text(DEFAULT_RESUME)
        resume_paras = get_resume_paragraphs(DEFAULT_RESUME)
    except Exception as pre_err:
        return jsonify({"error": f"Failed to read resume: {pre_err}"}), 500
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
                "content-type": "application/json",
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

    raw = strip_think(raw)

    replacements = extract_json_array(raw)
    role_suggestions = None
    if replacements is not None:
        role_suggestions, replacements = extract_role_suggestion(replacements)
        replacements = normalize_replacement_keys(replacements)
        replacements = filter_replacements_by_type(replacements, resume_paras)
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


def _tailor_openai_compatible(data):
    provider = data.get("provider", "lmstudio").lower().strip()
    jd_text = data.get("jd_text", "").strip()
    model = data.get("model", "").strip()
    job_title = data.get("job_title", "").strip()
    api_key = data.get("api_key", "").strip()
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
        return jsonify({"error": "Job description cannot be empty."}), 400
    if not os.path.exists(DEFAULT_RESUME):
        return jsonify({"error": "No resume found - please upload one first."}), 400

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

    # Pre-validate: parse docx and ensure output folder is writable before spending API credits
    try:
        resume_text = extract_resume_text(DEFAULT_RESUME)
        resume_paras = get_resume_paragraphs(DEFAULT_RESUME)
    except Exception as pre_err:
        return jsonify({"error": f"Failed to read resume: {pre_err}"}), 500
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

    raw = '[{"original":' if provider in local_providers else ""
    finish_reason = None
    stream_error = None
    try:
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line.startswith("event:"):
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line.startswith(
                ":"
            ):  # SSE comment line (e.g. ": OPENROUTER PROCESSING")
                continue
            if line.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(line)
                if "error" in chunk and "choices" not in chunk:
                    err_msg = chunk["error"]
                    if isinstance(err_msg, dict):
                        err_msg = err_msg.get("message", str(err_msg))
                    stream_error = str(err_msg)
                    print(f"[ERROR] Stream returned error: {stream_error}")
                    break
                choice = chunk["choices"][0]
                delta = choice["delta"].get("content", "")
                raw += delta
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            except Exception as error:
                print(f"[WARN] Failed to parse stream chunk: {line} - Error: {error}")
                continue
    except Exception as error:
        print(f"[ERROR] Stream reading failed: {error}")
        return jsonify({"error": f"Error reading stream: {error}"}), 500
    finally:
        if response:
            response.close()

    if stream_error:
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

    raw = strip_think(raw)
    print("[DEBUG] Raw output (first 800):", raw[:800])

    replacements = extract_json_array(raw)
    role_suggestions = None
    if replacements is not None:
        role_suggestions, replacements = extract_role_suggestion(replacements)
        replacements = normalize_replacement_keys(replacements)
        replacements = filter_replacements_by_type(replacements, resume_paras)

    if not replacements:
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
                    "No markdown fences, no explanation.\n\n" + raw[:4000]
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
            repair_raw = repair_response.json()["choices"][0]["message"]["content"]
            repair_raw = strip_think(repair_raw)
            print("[DEBUG] Repair output (first 500):", repair_raw[:500])
            replacements = extract_json_array(repair_raw)
            if replacements is not None:
                _rs, replacements = extract_role_suggestion(replacements)
                if _rs and not role_suggestions:
                    role_suggestions = _rs
                replacements = normalize_replacement_keys(replacements)
                replacements = filter_replacements_by_type(replacements, resume_paras)
        except Exception as error:
            print(f"[WARN] Repair request failed: {error}")

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


def _tailor_gemini_impl(data):
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
        return jsonify({"error": "Job description cannot be empty."}), 400
    if not os.path.exists(DEFAULT_RESUME):
        return jsonify({"error": "No resume found - please upload one first."}), 400

    # Pre-validate: parse docx and ensure output folder is writable before spending API credits
    try:
        resume_text = extract_resume_text(DEFAULT_RESUME)
        resume_paras = get_resume_paragraphs(DEFAULT_RESUME)
    except Exception as pre_err:
        return jsonify({"error": f"Failed to read resume: {pre_err}"}), 500
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
                response_mime_type="application/json",
            )

        response = client.models.generate_content(
            model=model_name, contents=prompt, config=config
        )
        raw = response.text
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

    raw = strip_think(raw)
    print("[DEBUG][Gemini] Raw output (first 800):", raw[:800])

    replacements = extract_json_array(raw)
    role_suggestions = None
    if replacements is not None:
        role_suggestions, replacements = extract_role_suggestion(replacements)
        replacements = normalize_replacement_keys(replacements)
        replacements = filter_replacements_by_type(replacements, resume_paras)

    if not replacements:
        print("[INFO][Gemini] First parse failed, attempting repair request...")
        try:
            repair_config = genai_types.GenerateContentConfig(
                system_instruction="You are a JSON repair assistant. Output ONLY a valid JSON array, nothing else.",
                temperature=0.0,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            )
            repair_prompt = (
                "The following text was supposed to be a JSON array of objects with "
                '"original" and "replacement" keys, but it could not be parsed. '
                "Extract the data and return ONLY a valid JSON array. "
                "No markdown fences, no explanation.\n\n" + raw[:4000]
            )
            repair_response = client.models.generate_content(
                model=model_name, contents=repair_prompt, config=repair_config
            )
            repair_raw = strip_think(repair_response.text)
            print("[DEBUG][Gemini] Repair output (first 500):", repair_raw[:500])
            replacements = extract_json_array(repair_raw)
            if replacements is not None:
                _rs, replacements = extract_role_suggestion(replacements)
                if _rs and not role_suggestions:
                    role_suggestions = _rs
                replacements = normalize_replacement_keys(replacements)
                replacements = filter_replacements_by_type(replacements, resume_paras)
        except Exception as error:
            print(f"[WARN][Gemini] Repair request failed: {error}")

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


def strip_think(text):
    """
    Strip <think>...</think> blocks from Qwen3/DeepSeek output.
    Handles: complete blocks, orphaned </think>, truncated blocks.
    """
    if not text:
        return text
    if "<think>" in text and "</think>" in text:
        cleaned = (
            __import__("re")
            .sub(r"<think>.*?</think>", "", text, flags=__import__("re").DOTALL)
            .strip()
        )
        if cleaned:
            return cleaned
    if "</think>" in text and "<think>" not in text:
        after = text.split("</think>", 1)[1].strip()
        return after if after else text.split("</think>", 1)[0].strip()
    if "<think>" in text:
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
