# Resume Tailor

Local webapp that tailors your resume to a job description using LM Studio (Qwen or any model).

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Make sure LM Studio is running on localhost:1234
#    with your Qwen model loaded

# 3. Run the app
python app.py
```

Then open http://localhost:5000

## How it works

1. Paste your **JD system prompt** (your tailoring instructions to the model)
2. Paste the **job description**
3. Click **Tailor Resume**
4. Review the before/after diff of every changed paragraph
5. Download the tailored `.docx` — same formatting as your original, content updated

## File structure

```
resume-tailor/
├── app.py                  # Flask backend
├── requirements.txt
├── templates/
│   └── index.html          # Frontend (uses oat.ink UI library)
├── static/
│   ├── oat.min.css         # oat UI (local copy)
│   └── oat.min.js
├── resume/
│   └── base_resume.docx    # Your master resume (upload via UI or drop here)
└── outputs/                # Generated tailored resumes (auto-created)
```

## Notes

- The app preserves all original `.docx` formatting — only text content is changed
- `<think>` tags from Qwen3 reasoning models are automatically stripped
- Your system prompt is saved in browser localStorage between sessions
- Outputs are timestamped so you never overwrite a previous version
