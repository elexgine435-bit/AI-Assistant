# AI-Assistant
# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) friendliness and suggests specific improvements, powered by Google's Gemini Flash model.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for keyword matching
- Overall **ATS score** (70% AI analysis + 30% rule-based structure checks)
- Category breakdown, strengths, missing keywords, prioritized improvements and example rewrites

## Run locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Provide your Gemini API key (free key: https://aistudio.google.com/apikey) in one of three ways:

1. Type it into the app's sidebar, or
2. Set an environment variable: `export GEMINI_API_KEY="your-key"` (Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`), or
3. Create `.streamlit/secrets.toml` containing `GEMINI_API_KEY = "your-key"`

Then run:
```bash
streamlit run app.py
```

## Change the model
The default is `gemini-3.5-flash`. Change it in the sidebar, or set the `GEMINI_MODEL` environment variable. If Google renames or retires a model, use any current Flash model name from the Gemini API docs.

## Push to GitHub
1. Create an empty repository on github.com (no README, since you already have one).
2. In your project folder, create a `.gitignore` file with:
   ```
   venv/
   __pycache__/
   .streamlit/secrets.toml
   .env
   ```
   **Never commit your API key.**
3. Run:
   ```bash
   git init
   git add app.py README.md requirements.txt .gitignore
   git commit -m "Initial commit: ATS resume checker"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

## Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** and choose your repository, branch `main`, and main file `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
4. Click **Deploy**. You'll get a public URL in a minute or two. Every `git push` to `main` redeploys automatically.

## Notes
- Scanned or image-only resumes can't be read (ATS systems can't read them either). Use a text-based PDF or DOCX.
- The score is an estimate, since real ATS systems differ. Treat it as guidance.
- Resume text is sent to the Gemini API, so don't upload anything you aren't comfortable sharing.
