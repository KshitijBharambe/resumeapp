# Resume Tailor

Local Flask webapp that tailors your resume to a job description using LM Studio or cloud providers.

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Run the app
python app.py
```

Then open http://localhost:8000

## How it works

1. Upload a base `.docx` resume
2. Paste a job description or extract one from a URL
3. Choose a provider and model
4. Click **Tailor Resume**
5. Review the before/after diff of every changed paragraph
6. Download the tailored `.docx`

## File structure

```
resume-tailor/
├── app.py                  # Flask routes and app entrypoint
├── config.py               # Shared paths, provider metadata, system prompt
├── services/
│   ├── jd_extraction_service.py
│   ├── resume_service.py
│   └── tailor_service.py
├── pyproject.toml
├── templates/
│   └── index.html          # Single-page frontend with inline CSS and JS
├── static/
├── resume/
│   └── base_resume.docx    # Your master resume (upload via UI or drop here)
└── outputs/                # Generated tailored resumes (auto-created)
```

## Notes

- The app preserves all original `.docx` formatting — only text content is changed
- `<think>` tags from reasoning-model output are stripped before JSON parsing
- Outputs are timestamped so you never overwrite a previous version
