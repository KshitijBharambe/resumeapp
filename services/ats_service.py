import json
import os
import re

import requests
from flask import jsonify

from config import DEFAULT_RESUME, PROVIDER_BASE_URLS, PROVIDER_DISPLAY
from services.resume_service import extract_resume_text
from services.tailor_service import get_chat_url_and_headers, strip_think

JOB_DESCRIPTION_EMPTY_ERROR = "Job description cannot be empty."
NO_RESUME_ERROR = "No resume found — please upload one first."
JSON_CONTENT_TYPE = "application/json"

ATS_SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and resume coach. \
Analyze the given resume against the job description and return ONLY a raw JSON object — \
no markdown fences, no prose, no explanation before or after.

Return this exact JSON structure:
{
  "ats_score": <integer 0-100>,
  "scoring_factors": [
    {"factor": "Keyword Match",          "score": <0-100>, "weight": 35, "details": "<2-3 sentence assessment>"},
    {"factor": "Skills Alignment",       "score": <0-100>, "weight": 25, "details": "<2-3 sentence assessment>"},
    {"factor": "Experience Relevance",   "score": <0-100>, "weight": 20, "details": "<2-3 sentence assessment>"},
    {"factor": "Education & Credentials","score": <0-100>, "weight": 10, "details": "<2-3 sentence assessment>"},
    {"factor": "Format & Structure",     "score": <0-100>, "weight": 10, "details": "<2-3 sentence assessment>"}
  ],
  "keywords": {
    "matched": ["keyword1", "keyword2"],
    "missing": ["keyword1", "keyword2"],
    "bonus":   ["keyword1", "keyword2"]
  },
  "analysis": {
    "strong_points": ["point1", "point2", "point3"],
    "weak_points":   ["point1", "point2", "point3"],
    "what_to_add":   ["suggestion1", "suggestion2", "suggestion3"],
    "what_to_change":["suggestion1", "suggestion2", "suggestion3"],
    "overall_summary": "<2-3 sentence honest and specific assessment of fit for this role>"
  }
}

SCORING RULES:
- ats_score: weighted sum — (sum of score_i * weight_i) / 100, rounded to nearest integer.
- Keyword Match (35%): percentage of critical JD keywords/phrases present in the resume. Focus on technical terms, required tools, domain-specific language, and role titles.
- Skills Alignment (25%): degree of overlap between the skills listed in the resume and JD requirements. Consider both exact matches and functional equivalents.
- Experience Relevance (20%): how well work history maps to the JD's required role, seniority level, and industry domain.
- Education & Credentials (10%): degree level, field, and certifications aligned to JD requirements. Score 100 if JD has no specific requirements.
- Format & Structure (10%): ATS-parseability — bullet points and clear section headers are good; dense prose, tables, or multi-column layouts that break parsing are bad.

KEYWORD RULES:
- matched: important keywords (technical skills, tools, platforms, methodologies, industry terms) present in BOTH the JD and the resume. Return 10–25 items.
- missing: critical keywords from the JD that are ABSENT from the resume. Prioritize must-have skills, required technologies, and role-defining terms. Return 5–15 items.
- bonus:   keywords in the resume not in the JD but broadly relevant to the target role and industry. Return 3–10 items.

ANALYSIS RULES:
- strong_points:  3–5 specific, evidence-based strengths this candidate has for this role.
- weak_points:    3–5 specific gaps, mismatches, or concerns about this candidate for this role.
- what_to_add:    3–5 actionable items the candidate should add to strengthen their resume for this JD (skills, experiences, certifications, keywords).
- what_to_change: 3–5 specific changes to existing resume content (e.g. "Reframe X bullet to emphasise Y", "Change job title from A to B").
- overall_summary: 2–3 honest, specific sentences summarising fit for this exact role.

Output ONLY the JSON object. No other text before or after."""


# ── JSON extraction ────────────────────────────────────────────────────────────


def _clean_llm_text(text):
    """Strip markdown fences, smart quotes, and think blocks."""
    text = re.sub(r"```+(?:json|JSON)?\s*", "", text)
    text = re.sub(r"```+", "", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _try_parse_ats_object(text, start_idx):  # NOSONAR
    """Scan from start_idx to find a balanced JSON object containing 'ats_score'."""
    depth = 0
    for i, c in enumerate(text[start_idx:], start_idx):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start_idx : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "ats_score" in obj:
                        return obj
                except Exception:
                    pass
                return None
    return None


def _extract_ats_json(text):
    """Robustly extract and parse the ATS JSON object from LLM output."""
    if not text:
        return None
    text = _clean_llm_text(text)
    for start_idx in (i for i, c in enumerate(text) if c == "{"):
        result = _try_parse_ats_object(text, start_idx)
        if result is not None:
            return result
    return None


# ── Helpers ────────────────────────────────────────────────────────────────────


def _build_ats_message(resume_text, jd_text):
    return f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd_text}"


def _load_resume():
    if not os.path.exists(DEFAULT_RESUME):
        return None, jsonify({"error": NO_RESUME_ERROR}), 400
    try:
        text = extract_resume_text(DEFAULT_RESUME)
        return text, None, None
    except Exception as err:
        return None, jsonify({"error": f"Failed to read resume: {err}"}), 500


def _provider_values(data):
    provider = data.get("provider", "lmstudio").lower().strip()
    api_key = data.get("api_key", "").strip()
    lm_url = data.get("lm_url", "").strip()
    if provider == "lmstudio" and not api_key:
        api_key = os.environ.get("LM_API_TOKEN", "")
    return provider, api_key, lm_url


# ── Dispatcher ─────────────────────────────────────────────────────────────────


def ats_score_resume(data):
    """Route to the correct provider handler."""
    provider = data.get("provider", "lmstudio").lower().strip()
    if provider == "gemini":
        return _ats_gemini(data)
    if provider == "anthropic":
        return _ats_anthropic(data)
    return _ats_openai_compatible(data)


# ── Anthropic ──────────────────────────────────────────────────────────────────


def _ats_anthropic(data):
    api_key = data.get("api_key", "").strip()
    model = data.get("model", "claude-sonnet-4-6").strip()
    jd_text = data.get("jd_text", "").strip()
    max_tokens = int(data.get("max_tokens", 4096))

    if not api_key:
        return jsonify({"error": "Anthropic API key is required."}), 400
    if not jd_text:
        return jsonify({"error": JOB_DESCRIPTION_EMPTY_ERROR}), 400

    resume_text, err_resp, err_code = _load_resume()
    if err_resp:
        return err_resp, err_code

    message = _build_ats_message(resume_text, jd_text)
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": ATS_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": message}],
            },
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": JSON_CONTENT_TYPE,
            },
            timeout=180,
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
    except Exception as err:
        return jsonify({"error": f"Anthropic API error: {err}"}), 500

    raw = strip_think(raw)
    result = _extract_ats_json(raw)
    if result is None:
        return (
            jsonify(
                {
                    "error": "Could not parse ATS analysis from Anthropic response.",
                    "raw_preview": raw[:1000],
                }
            ),
            500,
        )
    return jsonify(result)


# ── Gemini ─────────────────────────────────────────────────────────────────────


def _build_gemini_config(genai_types, use_thinking, temperature, max_tokens):
    if use_thinking:
        return genai_types.GenerateContentConfig(
            system_instruction=ATS_SYSTEM_PROMPT,
            temperature=1,
            max_output_tokens=max_tokens,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=4096),
        )
    return genai_types.GenerateContentConfig(
        system_instruction=ATS_SYSTEM_PROMPT,
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type=JSON_CONTENT_TYPE,
    )


def _gemini_error_response(err, model_name):
    err_str = str(err)
    if "API_KEY_INVALID" in err_str or "api key" in err_str.lower() or "401" in err_str:
        return (
            jsonify({"error": "Invalid Gemini API key. Check aistudio.google.com."}),
            401,
        )
    if "quota" in err_str.lower() or "429" in err_str:
        return jsonify({"error": "Gemini API quota exceeded. Try again later."}), 429
    if "not found" in err_str.lower() or "404" in err_str:
        return jsonify({"error": f"Model '{model_name}' not found on Gemini."}), 404
    return jsonify({"error": f"Gemini API error: {err_str}"}), 500


def _ats_gemini(data):
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
    model_name = data.get("model", "gemini-2.0-flash").strip()
    jd_text = data.get("jd_text", "").strip()
    max_tokens = int(data.get("max_tokens", 4096))
    temperature = float(data.get("temperature", 0.3))

    if not api_key:
        return jsonify({"error": "Gemini API key is required."}), 400
    if not jd_text:
        return jsonify({"error": JOB_DESCRIPTION_EMPTY_ERROR}), 400

    resume_text, err_resp, err_code = _load_resume()
    if err_resp:
        return err_resp, err_code

    prompt = _build_ats_message(resume_text, jd_text)
    thinking_models = {"gemini-2.5-pro", "gemini-2.5-flash", "gemini-3"}
    use_thinking = any(model_name.startswith(prefix) for prefix in thinking_models)

    try:
        client = genai.Client(api_key=api_key)
        config = _build_gemini_config(
            genai_types, use_thinking, temperature, max_tokens
        )
        response = client.models.generate_content(
            model=model_name, contents=prompt, config=config
        )
        raw = response.text or ""
    except Exception as err:
        return _gemini_error_response(err, model_name)

    raw = strip_think(raw)
    result = _extract_ats_json(raw)
    if result is None:
        return (
            jsonify(
                {
                    "error": "Could not parse ATS analysis from Gemini response.",
                    "raw_preview": raw[:1000],
                }
            ),
            500,
        )
    return jsonify(result)


# ── OpenAI-compatible (LM Studio, Ollama, OpenAI, Groq, OpenRouter, Mistral, Custom) ──


def _ats_openai_compatible(data):
    provider, api_key, lm_url = _provider_values(data)
    jd_text = data.get("jd_text", "").strip()
    model = data.get("model", "").strip()
    max_tokens = int(data.get("max_tokens", 4096))
    temperature = float(data.get("temperature", 0.3))

    if not jd_text:
        return jsonify({"error": JOB_DESCRIPTION_EMPTY_ERROR}), 400

    resume_text, err_resp, err_code = _load_resume()
    if err_resp:
        return err_resp, err_code

    chat_url, extra_headers = get_chat_url_and_headers(provider, lm_url, api_key)
    provider_label = PROVIDER_DISPLAY.get(provider, provider)

    message = _build_ats_message(resume_text, jd_text)

    local_providers = {"lmstudio", "ollama", "custom"}
    cloud_providers = {"openai", "groq", "openrouter", "mistral"}

    if provider in local_providers:
        message = message + "\n/no_think"

    messages = [
        {"role": "system", "content": ATS_SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    # Prefill assistant response for local providers to guide JSON output
    if provider in local_providers:
        messages.append({"role": "assistant", "content": '{"ats_score":'})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if provider in cloud_providers:
        payload["response_format"] = {"type": "json_object"}
    if provider in local_providers:
        payload["repeat_penalty"] = float(data.get("repeat_penalty", 1.1))

    try:
        response = requests.post(
            chat_url, json=payload, headers=extra_headers, timeout=180
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            jsonify(
                {
                    "error": f"Cannot connect to {provider_label}. Verify the URL is reachable."
                }
            ),
            503,
        )
    except requests.exceptions.Timeout:
        return jsonify({"error": f"{provider_label} timed out (3 min)."}), 504
    except Exception as err:
        return jsonify({"error": f"{provider_label} error: {err}"}), 500

    # Restore the prefilled prefix for local providers
    if provider in local_providers:
        raw = '{"ats_score":' + raw

    print(f"[DEBUG][ATS] Raw output (first 600): {raw[:600]}")

    raw = strip_think(raw)
    result = _extract_ats_json(raw)
    if result is None:
        return (
            jsonify(
                {
                    "error": "Could not parse ATS analysis from model response.",
                    "hint": "The model may not have returned valid JSON. Try a different model or increase Max Tokens.",
                    "raw_preview": raw[:1000],
                }
            ),
            500,
        )
    return jsonify(result)
