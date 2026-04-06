import io
import json
import os
import re
import threading
import queue

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)
from werkzeug.utils import secure_filename

from config import BASE_DIR, DEFAULT_RESUME, ORIGINAL_RESUME_INFO, OUTPUT_FOLDER
from services.jd_extraction_service import extract_job_description
from services.resume_service import apply_title_changes, resume_info_data
from services.tailor_service import provider_models, read_docx_b64, tailor_resume


app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


def run_with_keepalive(func, *args, **kwargs):
    environ = dict(request.environ)
    environ["wsgi.input"] = io.BytesIO(b"")
    messages = queue.Queue()

    def worker():
        with app.request_context(environ):
            try:
                result = func(*args, **kwargs)
                response = result[0] if isinstance(result, tuple) else result
                body = (
                    response.get_data(as_text=True)
                    if hasattr(response, "get_data")
                    else json.dumps(response)
                )
                messages.put({"type": "done", "body": body})
            except Exception as error:
                messages.put(
                    {
                        "type": "error",
                        "body": json.dumps({"error": f"Internal Error: {error}"}),
                    }
                )

    thread = threading.Thread(target=worker)
    thread.start()

    def generate():
        yield " "
        while True:
            try:
                message = messages.get(timeout=15)
                yield message["body"]
                break
            except queue.Empty:
                yield " "

    return Response(stream_with_context(generate()), mimetype="application/json")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload-resume", methods=["POST"])
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file_handle = request.files["file"]
    safe_name = secure_filename(file_handle.filename or "")
    if not safe_name.lower().endswith(".docx"):
        return jsonify({"error": "Only .docx files are accepted"}), 400

    file_handle.save(DEFAULT_RESUME)
    with open(ORIGINAL_RESUME_INFO, "w", encoding="utf-8") as output:
        output.write(safe_name)
    return jsonify({"success": True, **resume_info_data()})


@app.route("/resume-info", methods=["GET"])
def resume_info():
    return jsonify(resume_info_data())


@app.route("/extract-jd", methods=["POST"])
def extract_jd():
    url = (request.json or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "URL is required."}), 400

    try:
        text, company = extract_job_description(url)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify({"text": text, "company": company})


@app.route("/provider-models", methods=["POST"])
def provider_models_route():
    return provider_models(request.json or {})


@app.route("/apply-role-changes", methods=["POST"])
def apply_role_changes():
    data = request.json or {}
    filename = os.path.basename(data.get("filename", "").strip())
    changes = data.get("changes", [])

    if not filename or not re.fullmatch(r"[\w\-]+\.docx", filename):
        return jsonify({"error": "Invalid filename"}), 400
    if not changes or not isinstance(changes, list):
        return jsonify({"error": "No changes provided"}), 400

    src_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(src_path):
        return jsonify({"error": "Source file not found — re-run tailoring first"}), 404

    # Build new filename: insert _titled before the timestamp portion
    new_filename = re.sub(r"(\.docx)$", "_titled.docx", filename)
    dst_path = os.path.join(OUTPUT_FOLDER, new_filename)

    try:
        apply_title_changes(src_path, dst_path, changes)
    except Exception as error:
        return jsonify({"error": f"Failed to apply role changes: {error}"}), 500

    return jsonify(
        {
            "success": True,
            "filename": new_filename,
            "docx_b64": read_docx_b64(dst_path),
        }
    )


@app.route("/tailor", methods=["POST"])
def tailor():
    # Call the tailoring function synchronously so the client receives
    # proper JSON responses and HTTP status codes (errors will surface
    # to the UI). The streaming helper was causing the frontend to not
    # receive exception bodies reliably.
    return tailor_resume(request.json or {})


@app.errorhandler(Exception)
def handle_exception(error):
    import traceback

    traceback.print_exc()
    return jsonify({"error": str(error)}), 500


@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    safe = os.path.basename(filename)
    if not re.fullmatch(r"[\w\-]+\.docx", safe):
        return jsonify({"error": "Invalid filename"}), 400

    path = os.path.join(OUTPUT_FOLDER, safe)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n  Resume Tailor -> http://{host}:{port}\n")
    app.run(host=host, port=port)
