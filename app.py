import os
import json
import shutil
import requests
import re
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from docx import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "resume")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")


def _get_chat_url_and_headers(provider, lm_url, api_key):
    """Return (chat_completions_url, extra_headers) for the given provider."""
    local_providers = {"lmstudio", "ollama", "custom"}
    if provider in local_providers:
        base = (lm_url or PROVIDER_BASE_URLS.get(provider, "http://localhost:1234")).rstrip("/")
    else:
        base = PROVIDER_BASE_URLS.get(provider, "").rstrip("/")
    
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://resume-tailor.onrender.com"
    return f"{base}/v1/chat/completions", headers


def _tailor_anthropic(data):
    """Handle resume tailoring via the Anthropic Messages API."""
    api_key = data.get("api_key", "").strip()
    model = data.get("model", "claude-sonnet-4-6").strip()
    system_prompt = data.get("system_prompt", "").strip()
    jd_text = data.get("jd_text", "").strip()
    job_title = data.get("job_title", "").strip()
    max_tokens = int(data.get("max_tokens", 8192))

    if not api_key:
        return jsonify({"error": "Anthropic API key is required."}), 400
    if not system_prompt:
        return jsonify({"error": "System prompt cannot be empty."}), 400
    if not jd_text:
        return jsonify({"error": "Job description cannot be empty."}), 400
    if not os.path.exists(DEFAULT_RESUME):
        return jsonify({"error": "No resume found - please upload one first."}), 400

    resume_text = extract_resume_text(DEFAULT_RESUME)
    resume_paras = get_resume_paragraphs(DEFAULT_RESUME)

    combined_message = f"""JOB DESCRIPTION:
---
{jd_text}
---

CURRENT RESUME (full text):
---
{resume_text}
---

RESUME PARAGRAPHS (indexed):
---
{json.dumps(resume_paras, indent=2)}
---

Analyze the JD against the resume. Output ONLY a raw JSON array of changes.
Use this exact format:
[
  {{"original": "exact paragraph text verbatim from resume", "replacement": "tailored version"}}
]
Only include paragraphs that actually change. If nothing needs changing, output: []"""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            json={"model": model, "max_tokens": max_tokens, "system": system_prompt,
                  "messages": [{"role": "user", "content": combined_message}]},
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            timeout=300,
        )
        if resp.status_code == 401:
            return jsonify({"error": "Invalid Anthropic API key. Check console.anthropic.com."}), 401
        if resp.status_code == 429:
            return jsonify({"error": "Anthropic API rate limit exceeded. Try again shortly."}), 429
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"]
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Anthropic API."}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Anthropic API timed out."}), 504
    except Exception as e:
        return jsonify({"error": f"Anthropic API error: {e}"}), 500

    raw = _strip_think(raw)
    _last_raw["text"] = raw
    _last_raw["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    replacements = extract_json_array(raw)
    if replacements is None:
        preview = raw[:1200] if raw else "(empty)"
        return jsonify({"error": "Could not extract JSON array from Anthropic response.",
                        "hint": "Model returned prose without a JSON array.",
                        "raw_preview": preview}), 500

    enriched = enrich_with_sections(replacements, resume_paras)
    try:
        out_name = make_output_name(job_title)
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        apply_replacements(DEFAULT_RESUME, out_path, replacements)
    except Exception as e:
        return jsonify({"error": f"Failed to write .docx: {e}"}), 500

    return jsonify({"success": True, "filename": out_name, "docx_b64": _read_docx_b64(out_path),
                    "changes_count": len(enriched), "changes": enriched})


def make_output_name(job_title, prefix="resume"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if job_title:
        slug = re.sub(r"[^a-z0-9]+", "-", job_title.lower().strip()).strip("-")[:30]
        return f"{prefix}_{slug}_{ts}.docx"
    return f"{prefix}_tailored_{ts}.docx"
DEFAULT_RESUME = os.path.join(UPLOAD_FOLDER, "base_resume.docx")
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234")
LM_STUDIO_URL = f"{LM_STUDIO_BASE_URL.rstrip('/')}/v1/chat/completions"
LM_STUDIO_MODELS_URL = f"{LM_STUDIO_BASE_URL.rstrip('/')}/v1/models"

# Base URLs for cloud/local providers
PROVIDER_BASE_URLS = {
    "lmstudio":   LM_STUDIO_BASE_URL,
    "ollama":     "http://localhost:11434",
    "openai":     "https://api.openai.com",
    "groq":       "https://api.groq.com/openai",
    "openrouter": "https://openrouter.ai/api",
    "mistral":    "https://api.mistral.ai",
}

PROVIDER_DISPLAY = {
    "lmstudio": "LM Studio", "ollama": "Ollama", "openai": "OpenAI",
    "anthropic": "Anthropic", "gemini": "Gemini", "groq": "Groq",
    "openrouter": "OpenRouter", "mistral": "Mistral", "custom": "Custom",
}

# Static model lists for providers without a listing API
ANTHROPIC_MODELS = [
    {"id": "claude-opus-4-6"},
    {"id": "claude-sonnet-4-6"},
    {"id": "claude-haiku-4-5-20251001"},
]

OPENAI_STATIC_MODELS = [
    {"id": "gpt-4o"}, {"id": "gpt-4o-mini"}, {"id": "gpt-4-turbo"},
    {"id": "o3-mini"}, {"id": "o4-mini"},
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Store last raw model output for debug endpoint (with 1-hour TTL)
_last_raw = {"text": "", "ts": "", "time": 0}
_last_raw_ttl = 3600  # 1 hour in seconds


def extract_json_array(text):
    """
    Very robust JSON array extractor. Handles:
    - Markdown fences (```json ... ```)
    - <think> blocks already stripped upstream, but handle any leftovers
    - Model outputting analysis/prose THEN the array
    - Trailing commentary after the array
    - Single JSON object (wraps in list)
    - Individual objects scattered in text (fallback)
    - Escaped quotes, smart quotes, common model artifacts
    """
    if not text:
        return None

    # 1. Strip all markdown fences
    text = re.sub(r"```+(?:json|JSON)?\s*", "", text)
    text = re.sub(r"```+", "", text)

    # 2. Replace smart/curly quotes with straight ones
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # 3. Strip any remaining <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # 4. Try parsing the whole thing as-is
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)]
        if isinstance(result, dict):
            return [result]
    except Exception:
        pass

    # 5. Find ALL [...] blocks and try each
    bracket_starts = [i for i, c in enumerate(text) if c == "["]
    for start in bracket_starts:
        # Walk from the last ] backwards to find valid JSON
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, list) and len(result) > 0:
                            items = [r for r in result if isinstance(r, dict)]
                            if items:
                                return items
                    except Exception:
                        pass
                    break

    # 6. Extract individual JSON objects {original: ..., replacement: ...}
    #    This handles models that output objects one by one without wrapping array
    objects = []
    brace_starts = [i for i, c in enumerate(text) if c == "{"]
    for start in brace_starts:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and (
                            "original" in obj or "replacement" in obj
                        ):
                            objects.append(obj)
                    except Exception:
                        pass
                    break
    if objects:
        return objects

    # 7. Last resort: regex extraction of key-value pairs
    pattern = r'["\']?original["\']?\s*:\s*["\']([^"\']+)["\'].*?["\']?replacement["\']?\s*:\s*["\']([^"\']+)["\']'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return [{"original": o.strip(), "replacement": r.strip()} for o, r in matches]

    return None


def extract_resume_text(path):
    doc = Document(path)
    lines = []
    for p in doc.paragraphs:
        if p.text.strip():
            lines.append(p.text.strip())
    return "\n".join(lines)


def get_resume_paragraphs(path):
    """
    Return paragraphs with their index, text, style, section, and paragraph_type.

    paragraph_type values:
      - "heading"   : section header (WORK EXPERIENCE, SKILLS, etc.)
      - "title"     : name/contact line or role+company+date line (not rewriteable)
      - "summary"   : the professional summary paragraph(s)
      - "skills"    : a skills/certifications line
      - "bullet"    : a substantive experience/project bullet (primary rewrite target)
      - "other"     : anything else short/uncategorized
    """
    doc = Document(path)
    result = []
    current_section = ""
    heading_styles = {"heading 1", "heading 2", "heading 3", "heading 4"}

    # Date/role title pattern: lines containing date ranges or role descriptors
    _date_pat = re.compile(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|"
        r"april|june|july|august|september|october|november|december|\d{4})",
        re.IGNORECASE,
    )
    # Skills section keywords
    _skills_section_names = {"skills", "certifications", "education", "cert"}

    in_summary = False
    in_skills = False

    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        if not text:
            continue
        style_lower = p.style.name.lower()

        # ── Detect section headings ──
        is_heading = (
            any(h in style_lower for h in heading_styles)
            or (len(text) < 50 and text.isupper())
            or style_lower in ("title", "subtitle")
        )
        if is_heading:
            current_section = text
            sec_lower = text.lower()
            in_summary = "summary" in sec_lower
            in_skills = any(k in sec_lower for k in _skills_section_names)
            result.append(
                {
                    "index": i,
                    "text": text,
                    "style": p.style.name,
                    "section": current_section,
                    "is_heading": True,
                    "paragraph_type": "heading",
                }
            )
            continue

        # ── Classify non-heading paragraphs ──
        word_count = len(text.split())

        # Name / contact line: very short, no bullet indicator, near top
        is_contact_or_name = (
            word_count <= 12
            and not text.startswith(("•", "-", "–", "*", "◦"))
            and (
                "@" in text
                or "linkedin" in text.lower()
                or "github" in text.lower()
                or "portfolio" in text.lower()
                or "phone" in text.lower()
                or re.search(r"\d{3}[\-\.\s]\d{3}", text)
            )
        )

        # Role/company/date title line: contains a date OR is short with role-like content
        # These are lines like "DevOps Intern   June 2024 - August 2024"
        # or "Sequretek Pvt. Ltd.  June 2024 - August 2024"
        is_title_line = (
            not is_contact_or_name
            and word_count <= 15
            and _date_pat.search(text)
            and not text.startswith(("•", "-", "–", "*", "◦"))
        )

        # Project title line: looks like "ProjectName - Subtitle Month Year - Month Year"
        # These have dashes and date-like content but are longer
        is_project_title = (
            not is_contact_or_name
            and not is_title_line
            and word_count <= 25
            and _date_pat.search(text)
            and " - " in text
            and not text.startswith(("•", "-", "–", "*", "◦"))
        )

        if is_contact_or_name:
            para_type = "title"
        elif is_title_line or is_project_title:
            para_type = "title"
        elif in_summary and word_count >= 15:
            para_type = "summary"
        elif in_skills:
            para_type = "skills"
        elif word_count >= 15:
            para_type = "bullet"
        else:
            para_type = "other"

        result.append(
            {
                "index": i,
                "text": text,
                "style": p.style.name,
                "section": current_section,
                "is_heading": False,
                "paragraph_type": para_type,
            }
        )

    return result


def normalize_text(s):
    """Normalize for fuzzy matching — strips, collapses whitespace, removes zero-width chars."""
    s = s.strip()
    s = re.sub(r"[​‌‍﻿ \t]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def apply_replacements(original_path, output_path, replacements):
    """
    Apply text replacements to a docx while preserving all formatting.

    Matching strategy (in order):
    1. Exact match after normalize_text()
    2. Fuzzy match (ratio >= 0.85) for cases where LLM slightly altered the original text

    Writing strategy:
    - Preserve the formatting (bold, italic, font, size) of the first run
    - Set first run to full new text, clear all subsequent runs
    - This keeps bullet formatting, indentation, and style intact
    """
    import difflib

    shutil.copy2(original_path, output_path)
    doc = Document(output_path)

    # Build normalized lookup: normalized_text -> new_text
    rep_map = {}  # normalized original -> replacement
    raw_map = {}  # normalized original -> raw original (for logging)
    for r in replacements:
        if not isinstance(r, dict):
            print(f"[WARN] Skipping non-dict: {repr(r)[:80]}")
            continue
        old = r.get("original", "").strip()
        new = r.get("replacement", "").strip()
        if old and new:
            rep_map[normalize_text(old)] = new
            raw_map[normalize_text(old)] = old
        else:
            print(f"[WARN] Missing keys in: {list(r.keys())}")

    if not rep_map:
        doc.save(output_path)
        return

    matched = set()

    def write_para(para, new_text):
        """Replace paragraph text while preserving run formatting."""
        if not para.runs:
            # No runs — use XML direct manipulation
            from docx.oxml.ns import qn

            for child in list(para._p):
                if child.tag.endswith("}r"):  # w:r run element
                    para._p.remove(child)
            # Add a plain run
            from docx.oxml import OxmlElement

            run_el = OxmlElement("w:r")
            text_el = OxmlElement("w:t")
            text_el.text = new_text
            text_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            run_el.append(text_el)
            para._p.append(run_el)
            return
        # Set first run to full new text, blank out the rest
        para.runs[0].text = new_text
        for run in para.runs[1:]:
            run.text = ""

    applied = 0
    for para in doc.paragraphs:
        norm = normalize_text(para.text)
        if not norm:
            continue

        # 1. Exact match
        if norm in rep_map and norm not in matched:
            print(f"[MATCH exact] {repr(norm[:60])}")
            write_para(para, rep_map[norm])
            matched.add(norm)
            applied += 1
            continue

        # 2. Fuzzy match — find best candidate
        best_key = None
        best_ratio = 0.0
        for key in rep_map:
            if key in matched:
                continue
            ratio = difflib.SequenceMatcher(None, norm, key).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = key

        if best_key and best_ratio >= 0.85:
            print(f"[MATCH fuzzy {best_ratio:.2f}] {repr(norm[:60])}")
            write_para(para, rep_map[best_key])
            matched.add(best_key)
            applied += 1

    print(f"[INFO] Applied {applied}/{len(rep_map)} replacements")

    # Warn about unmatched
    for key in rep_map:
        if key not in matched:
            print(f"[UNMATCHED] {repr(key[:80])}")

    doc.save(output_path)


ORIGINAL_RESUME_INFO = os.path.join(UPLOAD_FOLDER, "original_filename.txt")

def resume_info_data():
    if not os.path.exists(DEFAULT_RESUME):
        return {"exists": False}
    text = extract_resume_text(DEFAULT_RESUME)
    doc = Document(DEFAULT_RESUME)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    original_fn = "base_resume.docx"
    if os.path.exists(ORIGINAL_RESUME_INFO):
        with open(ORIGINAL_RESUME_INFO, "r", encoding="utf-8") as f:
            original_fn = f.read().strip()
            
    return {
        "exists": True,
        "paragraphs": len(paras),
        "words": len(text.split()),
        "filename": "base_resume.docx",
        "original_filename": original_fn,
        "paragraphs_text": paras,
    }


def enrich_with_sections(replacements, resume_paras):
    """Add section context to each replacement dict."""
    import difflib

    para_lookup = {
        normalize_text(p["text"]): p.get("section", "") for p in resume_paras
    }
    enriched = []
    for r in replacements:
        if not isinstance(r, dict):
            continue
        orig_norm = normalize_text(r.get("original", ""))
        section = para_lookup.get(orig_norm, "")
        if not section and orig_norm:
            best_key, best_ratio = None, 0.0
            for key, sec in para_lookup.items():
                ratio = difflib.SequenceMatcher(None, orig_norm, key).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key = key
            if best_key and best_ratio >= 0.8:
                section = para_lookup[best_key]
        enriched.append({**r, "section": section})
    return enriched


@app.route("/")
def index():
    has_default = os.path.exists(DEFAULT_RESUME)
    return render_template("index.html", has_default=has_default)


@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".docx"):
        return jsonify({"error": "Only .docx files are accepted"}), 400
    f.save(DEFAULT_RESUME)
    with open(ORIGINAL_RESUME_INFO, "w", encoding="utf-8") as out:
        out.write(f.filename)
    return jsonify({"success": True, **resume_info_data()})


@app.route("/resume-info")
def resume_info():
    return jsonify(resume_info_data())


@app.route("/models")
def get_models():
    lm_url = request.args.get("lm_url", "").strip()
    models_url = f"{lm_url.rstrip('/')}/v1/models" if lm_url else LM_STUDIO_MODELS_URL
    try:
        r = requests.get(models_url, timeout=4)
        raw = r.json().get("data", [])
        models = []
        for m in raw:
            compat = m.get("compatibility_type", "")
            quant = m.get("quantization", "")
            arch = m.get("arch", "")
            if compat == "mlx":
                platform = "mac"
            elif compat in ("gguf", "exl2"):
                platform = "desktop"
            else:
                platform = ""
            models.append({"id": m["id"], "platform": platform, "compat": compat,
                            "quant": quant, "arch": arch})
        return jsonify({"models": models, "online": True})
    except Exception:
        return jsonify({"models": [], "online": False})


@app.route("/provider-models", methods=["POST"])
def provider_models():
    """Unified model listing for all providers."""
    data = request.json or {}
    provider = data.get("provider", "lmstudio").lower().strip()
    api_key = data.get("api_key", "").strip()
    lm_url = data.get("lm_url", "").strip()

    if provider == "lmstudio" and not api_key:
        api_key = os.environ.get("LM_API_TOKEN", "")

    # --- Anthropic: static list ---
    if provider == "anthropic":
        return jsonify({"models": ANTHROPIC_MODELS, "static": True})

    # --- Gemini: use existing google-genai listing ---
    if provider == "gemini":
        if not api_key:
            return jsonify({"error": "API key required"}), 400
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            raw = list(client.models.list())
            models = []
            for m in raw:
                name = m.name.split("/")[-1] if "/" in m.name else m.name
                if not any(x in name for x in ["gemini", "learnlm"]):
                    continue
                if "embedding" in name or "vision" in name:
                    continue
                tok = getattr(m, "input_token_limit", None) or getattr(m, "inputTokenLimit", None)
                models.append({"id": name, "input_token_limit": tok})
            models.sort(key=lambda x: x["id"])
            return jsonify({"models": models})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # --- Local providers (LM Studio, Ollama, Custom) ---
    if provider in ("lmstudio", "ollama", "custom"):
        base = (lm_url or PROVIDER_BASE_URLS.get(provider, "http://localhost:1234")).rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Ollama: try /api/tags first
        if provider == "ollama":
            try:
                r = requests.get(f"{base}/api/tags", headers=headers, timeout=4)
                r.raise_for_status()
                raw = r.json().get("models", [])
                models = [{"id": m["name"]} for m in raw]
                return jsonify({"models": models, "online": True})
            except Exception:
                pass  # fall through to OpenAI-compat endpoint
        try:
            r = requests.get(f"{base}/v1/models", headers=headers, timeout=4)
            r.raise_for_status()
            raw = r.json().get("data", [])
            models = []
            for m in raw:
                compat = m.get("compatibility_type", "")
                platform = "mac" if compat == "mlx" else ("desktop" if compat in ("gguf", "exl2") else "")
                models.append({"id": m["id"], "platform": platform,
                                "compat": compat, "quant": m.get("quantization", "")})
            return jsonify({"models": models, "online": True})
        except Exception:
            return jsonify({"models": [], "online": False})

    # --- OpenAI-compatible cloud providers ---
    base = PROVIDER_BASE_URLS.get(provider, "").rstrip("/")
    if not base:
        return jsonify({"error": f"Unknown provider: {provider}"}), 400
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if not api_key:
        return jsonify({"error": "API key required"}), 400
    try:
        r = requests.get(f"{base}/v1/models", headers=headers, timeout=8)
        if r.status_code == 401:
            return jsonify({"error": "Invalid API key"}), 401
        r.raise_for_status()
        raw = r.json().get("data", [])
        # Filter to chat-capable models
        chat_keywords = ["gpt", "claude", "llama", "mistral", "mixtral", "gemma", "qwen",
                         "deepseek", "command", "sonar", "hermes", "nous"]
        models = []
        for m in raw:
            mid = m.get("id", "")
            if any(k in mid.lower() for k in chat_keywords) or provider in ("groq", "mistral"):
                models.append({"id": mid})
        if not models:
            models = [{"id": m.get("id", "")} for m in raw]
        models.sort(key=lambda x: x["id"])
        return jsonify({"models": models})
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"Cannot connect to {PROVIDER_DISPLAY.get(provider, provider)}"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_with_keepalive(func, *args, **kwargs):
    import queue
    import threading
    from flask import current_app, request, Response, stream_with_context
    import json

    app = current_app._get_current_object()
    environ = dict(request.environ)
    import io
    environ["wsgi.input"] = io.BytesIO(b"")
    q = queue.Queue()

    def worker():
        with app.request_context(environ):
            try:
                res = func(*args, **kwargs)
                if isinstance(res, tuple):
                    resp = res[0]
                else:
                    resp = res
                body = resp.get_data(as_text=True) if hasattr(resp, "get_data") else json.dumps(resp)
                q.put({"type": "done", "body": body})
            except Exception as e:
                q.put({"type": "error", "body": json.dumps({"error": f"Internal Error: {str(e)}"})})

    t = threading.Thread(target=worker)
    t.start()

    def generate():
        yield " "  # Immediate whitespace to start transfer and bypass 100s TTFB
        while True:
            try:
                msg = q.get(timeout=15)
                yield msg["body"]
                break
            except queue.Empty:
                yield " "

    return Response(stream_with_context(generate()), mimetype='application/json')



@app.route("/tailor", methods=["POST"])
def tailor():
    data = request.json or {}
    return run_with_keepalive(_tailor_impl, data)

def _tailor_impl(data):
    provider = data.get("provider", "lmstudio").lower().strip()

    # Route to specialized handlers
    if provider == "gemini":
        return _tailor_gemini_impl(data)
    if provider == "anthropic":
        return _tailor_anthropic(data)

    # OpenAI-compatible path: lmstudio, ollama, openai, groq, openrouter, mistral, custom
    system_prompt = data.get("system_prompt", "").strip()
    jd_text = data.get("jd_text", "").strip()
    model = data.get("model", "").strip()
    job_title = data.get("job_title", "").strip()
    api_key = data.get("api_key", "").strip()
    lm_url = data.get("lm_url", "").strip()

    if provider == "lmstudio" and not api_key:
        api_key = os.environ.get("LM_API_TOKEN", "")

    # Advanced config (all optional)
    temperature = float(data.get("temperature", 0.3))
    max_tokens = int(data.get("max_tokens", 4096))
    top_p = float(data.get("top_p", 0.95))
    top_k = data.get("top_k")
    repeat_penalty = float(data.get("repeat_penalty", 1.1))
    seed = int(data.get("seed", -1))
    context_length = data.get("context_length")

    if not system_prompt:
        return jsonify({"error": "System prompt cannot be empty."}), 400
    if not jd_text:
        return jsonify({"error": "Job description cannot be empty."}), 400
    if not os.path.exists(DEFAULT_RESUME):
        return jsonify({"error": "No resume found - please upload one first."}), 400

    chat_url, extra_headers = _get_chat_url_and_headers(provider, lm_url, api_key)
    provider_label = PROVIDER_DISPLAY.get(provider, provider)

    resume_text = extract_resume_text(DEFAULT_RESUME)
    resume_paras = get_resume_paragraphs(DEFAULT_RESUME)

    combined_message = f"""JOB DESCRIPTION:
---
{jd_text}
---

CURRENT RESUME (full text):
---
{resume_text}
---

RESUME PARAGRAPHS (indexed):
---
{json.dumps(resume_paras, indent=2)}
---

Analyze the JD against the resume. Identify which paragraphs need tailoring to match the JD.
Apply all rules from your instructions (XYZ format, word limits, keyword injection, etc.).

Output ONLY a JSON array of changes — nothing else. No explanation, no markdown, no preamble.
Use this exact format:
[
  {{"original": "exact paragraph text verbatim from resume", "replacement": "tailored version"}}
]
Only include paragraphs that actually change. If nothing needs changing, output: []
Begin the JSON array now:"""

    local_providers = {"lmstudio", "ollama", "custom"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_message},
            {"role": "assistant", "content": "["},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": True,
    }
    if provider in local_providers:
        payload["repeat_penalty"] = repeat_penalty
        if top_k and int(top_k) > 0:
            payload["top_k"] = int(top_k)
        if seed != -1:
            payload["seed"] = seed
        if context_length:
            payload["context_length"] = int(context_length)

    resp = None
    try:
        resp = requests.post(chat_url, json=payload, headers=extra_headers, timeout=600, stream=True)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] ConnectionError connecting to {provider_label}: {e}")
        return jsonify({"error": f"Cannot connect to {provider_label}. Check the URL is reachable."}), 503
    except requests.exceptions.Timeout as e:
        print(f"[ERROR] Timeout connecting to {provider_label}: {e}")
        return jsonify({"error": f"{provider_label} timed out (10 min). Model may be too slow."}), 504
    except Exception as e:
        print(f"[ERROR] Exception connecting to {provider_label}: {e}")
        return jsonify({"error": f"{provider_label} error: {e}"}), 500

    raw = "["
    try:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"].get("content", "")
                raw += delta
            except Exception as e:
                print(f"[WARN] Failed to parse stream chunk: {line} - Error: {e}")
                continue
    except Exception as e:
        print(f"[ERROR] Stream reading failed: {e}")
        return jsonify({"error": f"Error reading stream: {e}"}), 500
    finally:
        if resp:
            resp.close()

    raw = _strip_think(raw)
    # Cache raw output with TTL
    current_time = time.time()
    if current_time - _last_raw.get("time", 0) > _last_raw_ttl:
        _last_raw["text"] = ""  # Clear expired cache
    _last_raw["text"] = raw
    _last_raw["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _last_raw["time"] = current_time
    print("[DEBUG] Raw output (first 800):", raw[:800])

    replacements = extract_json_array(raw)
    if replacements is None:
        preview = raw[:1200] if raw else "(empty — model produced no output)"
        return (
            jsonify(
                {
                    "error": "Could not extract a JSON array from the model response.",
                    "hint": "The model may have returned analysis/prose without a JSON array. Check the raw output below.",
                    "raw_preview": preview,
                }
            ),
            500,
        )

    enriched = enrich_with_sections(replacements, resume_paras)

    try:
        out_name = make_output_name(job_title)
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        apply_replacements(DEFAULT_RESUME, out_path, replacements)
    except Exception as e:
        return jsonify({"error": f"Failed to write .docx: {e}"}), 500

    return jsonify(
        {
            "success": True,
            "filename": out_name,
            "docx_b64": _read_docx_b64(out_path),
            "changes_count": len(enriched),
            "changes": enriched,
        }
    )


@app.route("/gemini-models", methods=["POST"])
def gemini_models():
    try:
        from google import genai
    except ImportError:
        return (
            jsonify(
                {
                    "error": "google-genai is not installed. Run: pip install google-genai"
                }
            ),
            500,
        )

    data = request.json or {}
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "API key required"}), 400

    try:
        client = genai.Client(api_key=api_key)
        pager = client.models.list(config={"page_size": 100})

        models = []
        for m in pager:
            actions = m.supported_actions or []
            if "generateContent" not in actions:
                continue
            name = m.name or ""
            model_id = (
                name.replace("models/", "") if name.startswith("models/") else name
            )
            if not model_id or "embedding" in model_id or "aqa" in model_id:
                continue
            models.append(
                {
                    "id": model_id,
                    "display_name": m.display_name or model_id,
                    "input_token_limit": m.input_token_limit,
                    "output_token_limit": m.output_token_limit,
                }
            )

        def sort_key(m):
            mid = m["id"]
            if "flash" in mid:
                return (0, mid)
            if "pro" in mid:
                return (1, mid)
            return (2, mid)

        models.sort(key=sort_key)
        return jsonify({"models": models})

    except Exception as e:
        err_str = str(e)
        if (
            "401" in err_str
            or "api key" in err_str.lower()
            or "invalid" in err_str.lower()
        ):
            return jsonify({"error": "Invalid API key"}), 401
        return jsonify({"error": f"Failed to fetch models: {err_str}"}), 500


@app.route("/tailor-gemini", methods=["POST"])
def tailor_gemini():
    data = request.json or {}
    return run_with_keepalive(_tailor_gemini_impl, data)

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
    system_prompt = data.get("system_prompt", "").strip()
    jd_text = data.get("jd_text", "").strip()
    job_title = data.get("job_title", "").strip()
    temperature = float(data.get("temperature", 0.3))
    max_tokens = int(data.get("max_tokens", 4096))

    top_p = float(data.get("top_p", 0.95))

    if not api_key:
        return jsonify({"error": "Gemini API key is required."}), 400
    if not system_prompt:
        return jsonify({"error": "System prompt cannot be empty."}), 400
    if not jd_text:
        return jsonify({"error": "Job description cannot be empty."}), 400
    if not os.path.exists(DEFAULT_RESUME):
        return jsonify({"error": "No resume found - please upload one first."}), 400

    resume_text = extract_resume_text(DEFAULT_RESUME)
    resume_paras = get_resume_paragraphs(DEFAULT_RESUME)

    prompt = f"""JOB DESCRIPTION:
---
{jd_text}
---

RESUME (full text):
---
{resume_text}
---

RESUME PARAGRAPHS (for output — copy "original" values verbatim from here):
---
{json.dumps(resume_paras, indent=2)}
---

Follow these steps internally, in order:

STEP 1 — KEYWORD EXTRACTION
List every keyword, skill, tool, framework, and theme the JD requires or rewards. Include both explicit requirements and implied ones (e.g. if it says "millions of concurrent users", the keyword is "scalable" / "high-traffic" / "concurrent").

STEP 2 — GAP ANALYSIS
For each keyword from Step 1, check whether it appears anywhere in the resume (any section). Mark it PRESENT or MISSING. Be strict — synonyms only count if they are close enough to satisfy an ATS.

STEP 3 — KEYWORD PLACEMENT
For each MISSING keyword, identify the single best paragraph in the resume to absorb it naturally. A keyword must fit the existing context of that paragraph — do not force it. If no paragraph can absorb it naturally, skip it.

STEP 4 — REWRITE
Rewrite only the paragraphs identified in Step 3, plus any paragraphs whose wording is genuinely weak for this role. Each rewrite must:
- Embed the target keyword(s) naturally in context
- Start with a varied, strong past-tense action verb
- Preserve all existing numbers and metrics exactly
- Stay under 32 words (bullets) or 50 words (summary)
- Sound like a human wrote it — varied structure, no repeated patterns

STEP 5 — OUTPUT
Output ONLY this JSON array, nothing else:
[
  {{"original": "exact paragraph text copied verbatim from the resume", "replacement": "rewritten version"}}
]
Only include paragraphs that actually changed. If nothing changed, output: []"""

    try:
        client = genai.Client(api_key=api_key)

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1,  # required for thinking mode
            max_output_tokens=max_tokens,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=8192),
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        raw = response.text

    except Exception as e:
        err_str = str(e)
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

    raw = _strip_think(raw)
    _last_raw["text"] = raw
    _last_raw["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[DEBUG][Gemini] Raw output (first 800):", raw[:800])

    replacements = extract_json_array(raw)
    if replacements is None:
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
        out_name = make_output_name(job_title)
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        apply_replacements(DEFAULT_RESUME, out_path, replacements)
    except Exception as e:
        return jsonify({"error": f"Failed to write .docx: {e}"}), 500

    return jsonify(
        {
            "success": True,
            "filename": out_name,
            "docx_b64": _read_docx_b64(out_path),
            "changes_count": len(enriched),
            "changes": enriched,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _strip_think(text):
    """
    Strip <think>...</think> blocks from Qwen3/DeepSeek output.
    Handles: complete blocks, orphaned </think>, truncated blocks.
    """
    if not text:
        return text
    # Case 1: complete block — strip it, keep what follows
    if "<think>" in text and "</think>" in text:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if cleaned:
            return cleaned
    # Case 2: orphaned </think> — prefill consumed the opener
    # Everything before </think> is thinking; everything after is the answer
    if "</think>" in text and "<think>" not in text:
        after = text.split("</think>", 1)[1].strip()
        return after if after else text.split("</think>", 1)[0].strip()
    # Case 3: <think> no </think> — truncated, fish out any JSON
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
    # Case 4: clean
    return text.strip()


def lm_call(
    model,
    system_prompt,
    user_message,
    temperature=0.15,
    max_tokens=2048,
    top_k=None,
    repeat_penalty=1.1,
    seed=-1,
    context_length=None,
    prefill=None,
    lm_url=None,
):
    """
    Single non-streaming call to LM Studio.
    Returns the raw text response or raises an exception.
    Optionally inject assistant prefill.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty,
        "stream": True,
    }
    if top_k and int(top_k) > 0:
        payload["top_k"] = int(top_k)
    if seed != -1:
        payload["seed"] = seed
    if context_length:
        payload["context_length"] = int(context_length)

    _chat_url = f"{lm_url.rstrip('/')}/v1/chat/completions" if lm_url else LM_STUDIO_URL
    resp = requests.post(_chat_url, json=payload, timeout=300, stream=True)
    resp.raise_for_status()

    raw = prefill or ""
    try:
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"].get("content", "")
                raw += delta
            except Exception:
                continue
    finally:
        resp.close()

    raw_pre = raw
    raw = _strip_think(raw)
    if not raw and raw_pre:
        print(
            f"[LM_CALL] WARNING: _strip_think emptied response. Pre-strip (first 200): {repr(raw_pre[:200])}"
        )
    else:
        print(f"[LM_CALL] Response OK, first 200: {repr(raw[:200])}")
    return raw


def lm_call_with_retry(
    model,
    system_prompt,
    user_message,
    parse_fn,
    temperature=0.15,
    max_tokens=2048,
    retries=2,
    **kwargs,
):
    """
    Call LM Studio and parse the result with parse_fn.
    Retries up to `retries` times on parse failure.
    Returns (result, raw_text) or raises on final failure.
    """
    # Append /no_think to suppress Qwen3 extended thinking for structured JSON tasks.
    # This is a Qwen3-specific token that disables the <think> block.
    # For other models it is ignored as unknown text at end of message.
    if "/no_think" not in user_message:
        user_message = user_message + "\n/no_think"

    last_raw = ""
    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = lm_call(
                model,
                system_prompt,
                user_message,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            last_raw = raw
            result = parse_fn(raw)
            if result is not None:
                return result, raw
            last_err = "parse_fn returned None"
        except Exception as e:
            last_err = str(e)
            last_raw = ""
        if attempt < retries:
            print(
                f"[PIPELINE] Retry {attempt + 1}/{retries} — {last_err}. Raw preview: {repr(last_raw[:200])}"
            )
    raise ValueError(
        f"Failed after {retries + 1} attempts. Last error: {last_err}\nRaw: {last_raw[:400]}"
    )


def gemini_call(
    api_key, model_name, system_prompt, user_message, temperature=0.4, max_tokens=512
):
    """
    Single Gemini API call. Returns raw text or raises.
    """
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config=config,
    )
    raw = response.text or ""
    raw = _strip_think(raw)
    return raw


def gemini_call_with_retry(
    api_key,
    model_name,
    system_prompt,
    user_message,
    parse_fn,
    temperature=0.4,
    max_tokens=512,
    retries=2,
):
    """Gemini call with retry and parse."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = gemini_call(
                api_key,
                model_name,
                system_prompt,
                user_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = parse_fn(raw)
            if result is not None:
                return result, raw
            last_err = "parse_fn returned None"
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            print(f"[PIPELINE][Gemini] Retry {attempt + 1}/{retries} — {last_err}")
    raise ValueError(
        f"Gemini failed after {retries + 1} attempts. Last error: {last_err}"
    )


# ─── STEP 1: JD ANALYSIS ───────────────────────────────────────────────────

STEP1_SYSTEM = """You are a job description analyst. Your ONLY job is to extract structured data from a JD and compare it to a resume.
DO NOT rewrite anything. DO NOT explain. Output JSON only — no markdown, no preamble, no backticks.
Example output shape:
{"role_type":"DevOps Engineer","keywords":["Terraform","Docker","CI/CD","AWS","Kubernetes","GitHub Actions","Python","IaC","monitoring","security"],"skill_gaps":["Kubernetes","FinOps"],"seniority":"mid","domain":"SaaS"}
/no_think"""


def run_step1(model, jd_text, resume_text, lm_kwargs):
    """Step 1: Analyze JD, extract keywords and skill gaps."""
    user_msg = f"""JOB DESCRIPTION:
{jd_text}

RESUME TEXT:
{resume_text}

Output a JSON object with these exact keys:
- role_type: string — primary role archetype
- keywords: array of top 10 ranked strings — most important JD keywords
- skill_gaps: array of strings — skills in JD not well-represented in resume
- seniority: string — junior/mid/senior/staff
- domain: string — industry domain (SaaS, FinTech, Healthcare, etc.)

Output JSON only. No explanation. No markdown."""

    def parse(raw):
        print(f"[STEP1 PARSE] raw length={len(raw)}, first 300: {repr(raw[:300])}")
        if not raw:
            return None
        raw = re.sub(r"```+(?:json)?", "", raw).strip("`").strip()
        # The response may start with { due to prefill — try direct parse first
        # Also search for any {...} block in case there's preamble
        candidates = [raw]
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m and m.group(0) != raw:
            candidates.append(m.group(0))
        for candidate in candidates:
            # Ensure it starts with {
            brace = candidate.find("{")
            if brace == -1:
                continue
            candidate = candidate[brace:]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "keywords" in obj and "role_type" in obj:
                    return obj
            except Exception:
                pass
            # Try finding the last } and trimming trailing garbage
            last_brace = candidate.rfind("}")
            if last_brace != -1:
                try:
                    obj = json.loads(candidate[: last_brace + 1])
                    if (
                        isinstance(obj, dict)
                        and "keywords" in obj
                        and "role_type" in obj
                    ):
                        return obj
                except Exception:
                    pass
        return None

    return lm_call_with_retry(
        model,
        STEP1_SYSTEM,
        user_msg,
        parse,
        temperature=0.15,
        max_tokens=2048,
        prefill="{",
        **lm_kwargs,
    )


# ─── STEP 2: CHANGE PLANNING ──────────────────────────────────────────

STEP2_SYSTEM = (
    "You are a resume change planner. Your ONLY job is to decide which resume paragraphs "
    "need tailoring for a specific JD.\n"
    "DO NOT rewrite anything. DO NOT explain. Output JSON array only.\n\n"
    "Paragraph types:\n"
    '  "bullet"  -> work/project bullet. Primary rewrite target. Use XYZ format.\n'
    '  "summary" -> professional summary. Rewrite to emphasize JD role + keywords.\n'
    '  "skills"  -> skills/certifications line. Front-load JD keywords.\n\n'
    'NEVER select paragraph_type "heading", "title", or "other".\n'
    "Select 4-8 paragraphs total. Spread across sections. "
    "Skip paragraphs that already strongly reflect the JD.\n\n"
    "Example output:\n"
    '[{"paragraph_index":5,"section":"WORK EXPERIENCE","paragraph_type":"bullet",'
    '"original_text":"Built REST APIs with FastAPI","reason":"inject CI/CD and Docker",'
    '"keywords_to_inject":["Docker","CI/CD"]},'
    '{"paragraph_index":2,"section":"SUMMARY","paragraph_type":"summary",'
    '"original_text":"Cloud-native engineer...","reason":"emphasize DevOps and GCP",'
    '"keywords_to_inject":["GCP","DevOps"]}]'
)


def run_step2(model, step1_result, resume_paras, lm_kwargs):
    """Step 2: Plan which paragraphs to change and what to inject."""
    # Only eligible types — never headings, titles, or short fragments
    eligible = [
        p
        for p in resume_paras
        if p.get("paragraph_type") in ("bullet", "summary", "skills")
        and len(p["text"]) > 20
    ]

    user_msg = (
        "JD ANALYSIS RESULT:\n"
        + json.dumps(step1_result, indent=2)
        + "\n\nRESUME PARAGRAPHS ELIGIBLE FOR REWRITING:\n"
        + json.dumps(eligible, indent=2)
        + "\n\nSelect 4-8 paragraphs that need tailoring to match the JD.\n"
        "For each, output:\n"
        "- paragraph_index: integer from the list\n"
        "- section: section name\n"
        '- paragraph_type: "bullet", "summary", or "skills"\n'
        "- original_text: exact text verbatim\n"
        "- reason: one sentence why it needs changing\n"
        "- keywords_to_inject: 2-3 JD keywords to weave in\n\n"
        "RULES:\n"
        '- ONLY select paragraph_type "bullet", "summary", or "skills"\n'
        "- Skip paragraphs that already strongly reflect the JD keywords\n"
        "- Spread changes across sections\n"
        "- Output JSON array only. No explanation."
    )

    def parse(raw):
        raw = re.sub(r"```+(?:json)?", "", raw).strip("`").strip()
        result = None
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list) and all("paragraph_index" in x for x in arr):
                    arr = [
                        x
                        for x in arr
                        if x.get("paragraph_type") in ("bullet", "summary", "skills")
                    ]
                    if arr:
                        result = arr
            except Exception:
                pass
        if result is None:
            objects = []
            for start in [idx for idx, c in enumerate(raw) if c == "{"]:
                depth = 0
                for i, c in enumerate(raw[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(raw[start : i + 1])
                                if "paragraph_index" in obj and obj.get(
                                    "paragraph_type"
                                ) in ("bullet", "summary", "skills"):
                                    objects.append(obj)
                            except Exception:
                                pass
                            break
            if objects:
                result = objects
        return result if result else None

    return lm_call_with_retry(
        model,
        STEP2_SYSTEM,
        user_msg,
        parse,
        temperature=0.15,
        max_tokens=2048,
        prefill="[",
        **lm_kwargs,
    )


# ─── STEP 3: REWRITING (GEMINI) ────────────────────────────────────────────

STEP3_BULLET_SYSTEM = (
    "You are an expert resume bullet writer. Your ONLY job is to rewrite ONE experience bullet "
    "to better match a target JD while preserving the candidate's real achievements.\n"
    'XYZ format: "Accomplished [X] as measured by [Y] by doing [Z]"\n\n'
    "RULES:\n"
    "- MAX 32 words. Every word counts.\n"
    "- Start with a strong past-tense action verb.\n"
    "- PRESERVE all existing numbers, percentages, and metrics EXACTLY as they appear. "
    "Do NOT invent new metrics.\n"
    "- If the original has no metrics, do not fabricate any — describe impact qualitatively.\n"
    "- Inject the provided keywords naturally — do NOT stuff awkwardly.\n"
    "- If the bullet already contains the keywords and is well-structured, make minimal changes only.\n"
    "- FORBIDDEN verbs: Led, Managed, Worked, Helped, Assisted, Supported, Utilized, Responsible for.\n"
    "- DO NOT output explanation, preamble, or markdown. Output the bullet text ONLY."
)

STEP3_SUMMARY_SYSTEM = (
    "You are a resume summary writer. Your ONLY job is to rewrite a professional summary "
    "to better align with a target JD.\n\n"
    "RULES:\n"
    "- Keep it under 50 words.\n"
    "- Lead with the candidate's actual role archetype and years of experience if mentioned.\n"
    "- Naturally weave in the provided JD keywords — do not keyword-stuff.\n"
    "- PRESERVE the candidate's real experience claims. Do NOT invent technologies.\n"
    "- Do NOT use first person (I, My, We).\n"
    "- Output the summary text ONLY — no explanation, no quotes, no markdown."
)

STEP3_SKILLS_SYSTEM = (
    "You are a resume skills section editor. Your ONLY job is to rewrite ONE skills line "
    "to better reflect a target JD.\n\n"
    "RULES:\n"
    "- Keep the same general format and length as the original.\n"
    "- Move JD-relevant skills to the front of the line.\n"
    "- Remove skills not relevant to this JD only if the line is overcrowded.\n"
    "- PRESERVE all real skills — do NOT invent skills the candidate does not have.\n"
    "- If the keywords are already present, make minimal or no changes.\n"
    "- Output the skills line text ONLY — no explanation, no markdown."
)


def run_step3_for_group(
    api_key,
    gemini_model,
    bullets_in_group,
    keywords_to_inject,
    section_name,
    used_verbs,
):
    """
    Rewrite all items in one section group via Gemini.
    Dispatches to the right system prompt based on paragraph_type.
    Updates used_verbs set for bullet type only.
    Returns list of result dicts.
    """
    results = []
    for bullet in bullets_in_group:
        para_type = bullet.get("paragraph_type", "bullet")

        if para_type == "summary":
            system = STEP3_SUMMARY_SYSTEM
            user_msg = (
                f"ORIGINAL SUMMARY: {bullet['original_text']}\n"
                f"JD KEYWORDS TO WEAVE IN: {', '.join(keywords_to_inject)}\n\n"
                "Rewrite the summary to better match this JD. Under 50 words. "
                "Output the summary text only."
            )
            max_tok = 120

        elif para_type == "skills":
            system = STEP3_SKILLS_SYSTEM
            user_msg = (
                f"ORIGINAL SKILLS LINE: {bullet['original_text']}\n"
                f"JD KEYWORDS TO FRONT-LOAD: {', '.join(keywords_to_inject)}\n\n"
                "Rewrite this skills line to lead with the most JD-relevant skills. "
                "Output the skills line text only."
            )
            max_tok = 150

        else:  # bullet
            verb_exclusion = ", ".join(sorted(used_verbs)) if used_verbs else "none yet"
            system = STEP3_BULLET_SYSTEM
            user_msg = (
                f"SECTION: {section_name}\n"
                f"ORIGINAL BULLET: {bullet['original_text']}\n"
                f"KEYWORDS TO INJECT: {', '.join(keywords_to_inject)}\n"
                f"ALREADY-USED VERBS (DO NOT repeat): {verb_exclusion}\n\n"
                "Rewrite using XYZ format. MAX 32 words. "
                "Preserve all existing numbers and metrics exactly. "
                "Output the bullet text only."
            )
            max_tok = 180

        def make_parse(ptype):
            def parse(raw):
                raw = raw.strip().strip('"').strip()
                # Remove any leading bullet character that Gemini sometimes adds
                raw = re.sub(r"^[-•–\*]\s*", "", raw).strip()
                if raw and len(raw.split()) <= 80:
                    return raw
                return None

            return parse

        try:
            rewritten, _ = gemini_call_with_retry(
                api_key,
                gemini_model,
                system,
                user_msg,
                make_parse(para_type),
                temperature=0.4,
                max_tokens=max_tok,
                retries=2,
            )
            # Track verb for bullets only
            if para_type == "bullet" and rewritten:
                first_word = rewritten.split()[0].rstrip(",").lower()
                if first_word:
                    used_verbs.add(first_word)
            results.append(
                {
                    "original_text": bullet["original_text"],
                    "rewritten": rewritten,
                    "paragraph_index": bullet["paragraph_index"],
                    "paragraph_type": para_type,
                }
            )
        except Exception as e:
            print(
                f"[PIPELINE][Step3] Gemini failed for {para_type}, using original. Error: {e}"
            )
            results.append(
                {
                    "original_text": bullet["original_text"],
                    "rewritten": bullet["original_text"],
                    "paragraph_index": bullet["paragraph_index"],
                    "paragraph_type": para_type,
                }
            )
    return results


# ─── STEP 4: VALIDATION + FIX ───────────────────────────────────────────────

STEP4_SYSTEM = (
    "You are a resume content validator. Your ONLY job is to check a CANDIDATE (already-rewritten) "
    "resume text and fix any quality issues with it.\n\n"
    "You receive items with a 'candidate' field — this is the text to validate and output.\n"
    "NEVER revert to any older version. Always output the candidate text or an improved version.\n\n"
    'For BULLETS (paragraph_type="bullet"), flag and fix if:\n'
    "  1. Over 32 words\n"
    "  2. Missing XYZ structure (achievement + measure/impact + method)\n"
    "  3. Forbidden verbs: Led, Managed, Worked, Helped, Assisted, Supported, Utilized\n"
    "  4. Starts with a pronoun (I, We, My)\n"
    "  5. Contains fabricated metrics (numbers that weren't in the original)\n\n"
    'For SUMMARY (paragraph_type="summary"), flag and fix if:\n'
    "  1. Over 50 words\n"
    '  2. Starts with "I" or "My"\n\n'
    'For SKILLS (paragraph_type="skills"), flag and fix if:\n'
    "  1. Becomes significantly longer than typical skills lines\n\n"
    "Output JSON array only. No markdown, no explanation.\n"
    "Example:\n"
    '[{"original":"Architected REST APIs...","rewritten":"Architected REST APIs as measured by 30% latency drop '
    'by implementing async FastAPI","paragraph_type":"bullet","status":"fixed"},'
    '{"original":"Deployed Docker containers...","rewritten":"Deployed Docker containers...","paragraph_type":"bullet","status":"pass"}]\n/no_think'
)


def run_step4(model, rewritten_bullets, lm_kwargs):
    """Step 4: Validate all rewritten content, fix failures."""
    if not rewritten_bullets:
        return [], ""

    bullets_for_validation = [
        {
            "index": i,
            "candidate": b["rewritten"],
            "paragraph_type": b.get("paragraph_type", "bullet"),
        }
        for i, b in enumerate(rewritten_bullets)
    ]

    user_msg = (
        "Validate each CANDIDATE text below based on its paragraph_type rules.\n\n"
        "ITEMS TO VALIDATE:\n"
        + json.dumps(bullets_for_validation, indent=2)
        + "\n\nFor each item output:\n"
        "  original: copy the 'candidate' text exactly as-is\n"
        "  rewritten: if passing, copy the 'candidate' text exactly as-is; if fixing, output the corrected version\n"
        "  paragraph_type: same as input\n"
        '  status: "pass" or "fixed"\n\n'
        "CRITICAL: 'rewritten' must always be the candidate text or an improved version of it — NEVER revert to an older version.\n"
        "CRITICAL: Do NOT fabricate metrics. If a bullet has no numbers, keep it metric-free.\n"
        "Output JSON array only."
    )

    def parse(raw):
        raw = re.sub(r"```+(?:json)?", "", raw).strip("`").strip()
        arr = extract_json_array(raw)
        if arr and all(isinstance(x, dict) and "rewritten" in x for x in arr):
            return arr
        return None

    try:
        validated, raw = lm_call_with_retry(
            model,
            STEP4_SYSTEM,
            user_msg,
            parse,
            temperature=0.15,
            max_tokens=2048,
            prefill="[",
            **lm_kwargs,
        )
        for i, v in enumerate(validated):
            if i < len(rewritten_bullets):
                rewritten_bullets[i]["validated"] = v.get(
                    "rewritten", rewritten_bullets[i]["rewritten"]
                )
                rewritten_bullets[i]["validation_status"] = v.get("status", "unknown")
        return rewritten_bullets, raw
    except Exception as e:
        print(f"[PIPELINE][Step4] Validation failed: {e}. Using step3 output as-is.")
        for b in rewritten_bullets:
            b["validated"] = b["rewritten"]
            b["validation_status"] = "skipped"
        return rewritten_bullets, ""


# ─── STEP 5: JSON ASSEMBLY ──────────────────────────────────────────────


def run_step5(validated_bullets, resume_paras):
    """Step 5: Assemble final [{original, replacement}] array — pure Python, no model call."""
    para_lookup = {p["index"]: p["text"] for p in resume_paras}

    result = []
    for b in validated_bullets:
        idx = b.get("paragraph_index")
        # Use the exact paragraph text from the docx as the original (most reliable match)
        original_text = para_lookup.get(idx, b.get("original_text", ""))
        replacement_text = b.get(
            "validated", b.get("rewritten", b.get("original_text", ""))
        )
        print(
            f"[STEP5 DEBUG] idx={idx} para_hit={'yes' if idx in para_lookup else 'no'} "
            f"orig[:60]={repr((original_text or '')[:60])} "
            f"repl[:60]={repr((replacement_text or '')[:60])} "
            f"same={original_text.strip()==replacement_text.strip() if original_text and replacement_text else 'n/a'}"
        )
        if (
            original_text
            and replacement_text
            and original_text.strip() != replacement_text.strip()
        ):
            result.append({"original": original_text, "replacement": replacement_text})

    print(f"[PIPELINE][Step5] Assembled {len(result)} replacement pairs directly.")
    return result, ""


# ─── PIPELINE ENDPOINT ───────────────────────────────────────────────────────


@app.route("/tailor-pipeline", methods=["POST"])
def tailor_pipeline():
    data = request.json or {}
    return run_with_keepalive(_tailor_pipeline_impl, data)

def _tailor_pipeline_impl(data):
    """
    5-step pipeline:
    1. JD Analysis          → LM Studio (Qwen)
    2. Change Planning      → LM Studio (Qwen)
    3. Bullet Rewriting     → Gemini (per role/project group)
    4. Validation + Fix     → LM Studio (Qwen)
    5. JSON Assembly        → LM Studio (Qwen)
    """
    try:
        from google import genai
    except ImportError:
        return (
            jsonify(
                {
                    "error": "google-genai is not installed. Run: pip install google-genai"
                }
            ),
            500,
        )

    jd_text = data.get("jd_text", "").strip()
    api_key = data.get("api_key", "").strip()
    local_model = data.get("local_model", "").strip()
    gemini_model = data.get("gemini_model", "gemini-3.1-pro-preview").strip()
    pipeline_lm_url = data.get("lm_url", "").strip()

    # Advanced config for LM Studio steps (optional)
    top_k = data.get("top_k")
    repeat_penalty = float(data.get("repeat_penalty", 1.1))
    seed = int(data.get("seed", -1))
    context_length = data.get("context_length")

    lm_kwargs = {}
    if top_k and int(top_k) > 0:
        lm_kwargs["top_k"] = int(top_k)
    if seed != -1:
        lm_kwargs["seed"] = seed
    if context_length:
        lm_kwargs["context_length"] = int(context_length)
    lm_kwargs["repeat_penalty"] = repeat_penalty
    if pipeline_lm_url:
        lm_kwargs["lm_url"] = pipeline_lm_url

    if not jd_text:
        return jsonify({"error": "Job description cannot be empty."}), 400
    if not api_key:
        return jsonify({"error": "Gemini API key is required for the pipeline."}), 400
    if not local_model:
        return jsonify({"error": "No local model selected for pipeline steps."}), 400
    if not os.path.exists(DEFAULT_RESUME):
        return jsonify({"error": "No resume found — please upload one first."}), 400

    resume_text = extract_resume_text(DEFAULT_RESUME)
    resume_paras = get_resume_paragraphs(DEFAULT_RESUME)

    pipeline_log = []

    # ── STEP 1: JD Analysis ────────────────────────────────────────────────
    print("[PIPELINE] Step 1: JD Analysis")
    pipeline_log.append({"step": 1, "name": "JD Analysis", "status": "running"})
    try:
        step1_result, _ = run_step1(local_model, jd_text, resume_text, lm_kwargs)
        pipeline_log[-1]["status"] = "done"
        pipeline_log[-1][
            "result_preview"
        ] = f"role_type={step1_result.get('role_type')}, {len(step1_result.get('keywords', []))} keywords"
        print(
            f"[PIPELINE] Step 1 done: {step1_result.get('role_type')}, keywords: {step1_result.get('keywords', [])[:5]}"
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Step 1 (JD Analysis) failed: {e}",
                    "pipeline_log": pipeline_log,
                }
            ),
            500,
        )

    # ── STEP 2: Change Planning ────────────────────────────────────────────
    print("[PIPELINE] Step 2: Change Planning")
    pipeline_log.append({"step": 2, "name": "Change Planning", "status": "running"})
    try:
        step2_result, _ = run_step2(local_model, step1_result, resume_paras, lm_kwargs)
        pipeline_log[-1]["status"] = "done"
        pipeline_log[-1][
            "result_preview"
        ] = f"{len(step2_result)} paragraphs planned for rewrite"
        print(f"[PIPELINE] Step 2 done: {len(step2_result)} changes planned")
    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Step 2 (Change Planning) failed: {e}",
                    "pipeline_log": pipeline_log,
                }
            ),
            500,
        )

    # ── STEP 3: Bullet Rewriting (Gemini, grouped by section) ─────────────
    print("[PIPELINE] Step 3: Bullet Rewriting via Gemini")
    pipeline_log.append({"step": 3, "name": "Bullet Rewriting", "status": "running"})

    # Group planned changes by section
    from collections import defaultdict

    section_groups = defaultdict(list)
    for change in step2_result:
        section_groups[change.get("section", "Other")].append(change)

    used_verbs = set()
    all_rewritten = []

    try:
        for section_name, bullets in section_groups.items():
            # Collect keywords for this group (union of all bullets' keywords)
            group_keywords = []
            for b in bullets:
                for kw in b.get("keywords_to_inject", []):
                    if kw not in group_keywords:
                        group_keywords.append(kw)

            group_results = run_step3_for_group(
                api_key, gemini_model, bullets, group_keywords, section_name, used_verbs
            )
            all_rewritten.extend(group_results)
            print(
                f"[PIPELINE] Step 3: rewrote {len(bullets)} bullets for section '{section_name}'"
            )

        pipeline_log[-1]["status"] = "done"
        pipeline_log[-1]["result_preview"] = f"{len(all_rewritten)} bullets rewritten"
    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Step 3 (Bullet Rewriting) failed: {e}",
                    "pipeline_log": pipeline_log,
                }
            ),
            500,
        )

    # ── STEP 4: Validation + Fix ───────────────────────────────────────────
    print("[PIPELINE] Step 4: Validation + Fix")
    pipeline_log.append({"step": 4, "name": "Validation", "status": "running"})
    try:
        validated_bullets, _ = run_step4(local_model, all_rewritten, lm_kwargs)
        passed = sum(
            1 for b in validated_bullets if b.get("validation_status") == "pass"
        )
        fixed = sum(
            1 for b in validated_bullets if b.get("validation_status") == "fixed"
        )
        pipeline_log[-1]["status"] = "done"
        pipeline_log[-1]["result_preview"] = f"{passed} pass, {fixed} fixed"
        print(f"[PIPELINE] Step 4 done: {passed} pass, {fixed} fixed")
    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Step 4 (Validation) failed: {e}",
                    "pipeline_log": pipeline_log,
                }
            ),
            500,
        )

    # ── STEP 5: JSON Assembly ──────────────────────────────────────────────
    print("[PIPELINE] Step 5: JSON Assembly")
    pipeline_log.append({"step": 5, "name": "Assembly", "status": "running"})
    try:
        replacements, _ = run_step5(validated_bullets, resume_paras)
        pipeline_log[-1]["status"] = "done"
        pipeline_log[-1][
            "result_preview"
        ] = f"{len(replacements)} replacements assembled"
        print(f"[PIPELINE] Step 5 done: {len(replacements)} replacements")
    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"Step 5 (Assembly) failed: {e}",
                    "pipeline_log": pipeline_log,
                }
            ),
            500,
        )

    if not replacements:
        return jsonify(
            {
                "success": True,
                "filename": "",
                "changes_count": 0,
                "changes": [],
                "pipeline_log": pipeline_log,
                "step1_analysis": step1_result,
            }
        )

    # Enrich with section data
    enriched = enrich_with_sections(replacements, resume_paras)

    # Apply to docx
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"resume_pipeline_{ts}.docx"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        apply_replacements(DEFAULT_RESUME, out_path, replacements)
    except Exception as e:
        return (
            jsonify(
                {"error": f"Failed to write .docx: {e}", "pipeline_log": pipeline_log}
            ),
            500,
        )

    return jsonify(
        {
            "success": True,
            "filename": out_name,
            "docx_b64": _read_docx_b64(out_path),
            "changes_count": len(enriched),
            "changes": enriched,
            "pipeline_log": pipeline_log,
            "step1_analysis": step1_result,
        }
    )


@app.route("/raw-log")
def raw_log():
    return jsonify(
        {
            "ts": _last_raw.get("ts", ""),
            "length": len(_last_raw.get("text", "")),
            "raw": _last_raw.get("text", "(no output yet)"),
        }
    )


# Always return JSON, never an HTML error page
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback

    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    safe = os.path.basename(filename)
    path = os.path.join(OUTPUT_FOLDER, safe)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


def _read_docx_b64(path):
    """Read a .docx file and return it base64-encoded (for localStorage caching)."""
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _convert_docx_to_pdf(docx_path, pdf_path):
    """Convert a .docx to .pdf. Returns None on success, error string on failure.
    Tries (in order): LibreOffice headless, docx2pdf.
    Works on Windows, macOS, and Linux.
    """
    import subprocess, sys, shutil

    docx_abs = os.path.abspath(docx_path)
    pdf_abs = os.path.abspath(pdf_path)
    out_dir = os.path.dirname(pdf_abs)

    # --- Method 1: LibreOffice headless (cross-platform) ---
    soffice = None
    candidates = ["soffice", "libreoffice"]
    if sys.platform == "darwin":
        candidates.insert(0, "/Applications/LibreOffice.app/Contents/MacOS/soffice")
    elif sys.platform == "win32":
        import glob as _glob
        for g in _glob.glob(r"C:\Program Files\LibreOffice\program\soffice.exe"):
            candidates.insert(0, g)
    for c in candidates:
        if shutil.which(c) or os.path.exists(c):
            soffice = c
            break

    if soffice:
        try:
            import os as _os, uuid
            env = dict(_os.environ)
            env.setdefault("HOME", "/tmp")  # writable HOME required in containers
            profile_dir = f"/tmp/lo_profile_{uuid.uuid4().hex}"
            result = subprocess.run(
                [soffice, "--headless", "--norestore", "--nofirststartwizard",
                 f"-env:UserInstallation=file://{profile_dir}",
                 "--convert-to", "pdf", "--outdir", out_dir, docx_abs],
                capture_output=True, text=True, timeout=60, env=env
            )
            # LibreOffice outputs <basename>.pdf, rename if needed
            lo_out = os.path.join(out_dir, os.path.splitext(os.path.basename(docx_abs))[0] + ".pdf")
            if os.path.exists(lo_out) and lo_out != pdf_abs:
                os.rename(lo_out, pdf_abs)
            if os.path.exists(pdf_abs):
                return None
        except Exception:
            pass  # fall through to next method

    # --- Method 2: docx2pdf (uses Word on Windows/macOS) ---
    try:
        from docx2pdf import convert
        convert(docx_abs, pdf_abs)
        if os.path.exists(pdf_abs):
            return None
    except ImportError:
        pass
    except Exception as e:
        return str(e)

    return "No PDF converter available. Install LibreOffice (recommended) or ensure Microsoft Word is accessible."


@app.route("/download-pdf/<filename>")
def download_pdf(filename):
    safe = os.path.basename(filename)
    docx_path = os.path.join(OUTPUT_FOLDER, safe)
    if not os.path.exists(docx_path):
        return jsonify({"error": "File not found"}), 404
    pdf_name = safe.replace(".docx", ".pdf")
    pdf_path = os.path.join(OUTPUT_FOLDER, pdf_name)
    err = _convert_docx_to_pdf(docx_path, pdf_path)
    if err:
        return jsonify({"error": f"PDF conversion failed: {err}"}), 500
    if not os.path.exists(pdf_path):
        return jsonify({"error": "PDF conversion produced no output"}), 500
    return send_file(pdf_path, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  Resume Tailor -> http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port)
