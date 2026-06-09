# ATS Resume Builder — Setup & Deployment Guide

## Files
- `backend/main.py` — FastAPI backend
- `backend/requirements.txt` — Python dependencies
- `backend/Procfile` — for Railway deployment
- `index.html` — frontend (open in browser)
- `Junaid_Hasan_CV.tex` — your CV template

---

## Run Locally
```bash
cd backend
pip install -r requirements.txt
set GROQ_API_KEY=your_key_here
python -m uvicorn main:app --reload --port 8000
```
Then open `index.html` via `python -m http.server 3000` → http://localhost:3000

---

## Deploy Online (Railway + GitHub Pages)

### Step 1 — Push to GitHub
1. Go to github.com → New Repository → name it `ats-resume-builder` → Create
2. Open terminal in your project folder and run:
```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ats-resume-builder.git
git push -u origin main
```

### Step 2 — Deploy Backend on Railway
1. Go to railway.app → Login with GitHub
2. Click New Project → Deploy from GitHub repo → select `ats-resume-builder`
3. Set Root Directory to `backend`
4. Go to Variables tab → Add: `GROQ_API_KEY` = your key
5. Deploy — Railway gives you a URL like `https://ats-resume-builder.up.railway.app`

### Step 3 — Update Frontend
In `index.html`, find:
```js
const API = "http://localhost:8000";
```
Replace with your Railway URL:
```js
const API = "https://ats-resume-builder.up.railway.app";
```

### Step 4 — Deploy Frontend on GitHub Pages
1. Go to your GitHub repo → Settings → Pages
2. Source: Deploy from branch → main → / (root) → Save
3. Your frontend is live at: `https://YOUR_USERNAME.github.io/ats-resume-builder`
