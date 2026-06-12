"""
ATS Resume Builder Agent - FastAPI Backend (Groq)
Supports multiple named CV templates (slots)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, re
from groq import Groq

app = FastAPI(title="ATS Resume Builder Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# In-memory storage — slot-based, no disk needed
# slots: dict of slot_id -> {"title": str, "filename": str, "content": str}
templates_store = {}


# ─── Models ───────────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    job_description: str
    job_id: str = "job-1"
    slot_id: str


class SuggestRequest(BaseModel):
    job_description: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def chat(prompt: str, temperature: float = 0.3) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ATS Resume Builder Agent running", "model": MODEL}


@app.post("/upload-template")
async def upload_template(
    file: UploadFile = File(...),
    slot_id: str = Form(...),
    title: str = Form(...)
):
    if not file.filename.endswith(".tex"):
        raise HTTPException(status_code=400, detail="Only .tex files are accepted.")
    content = (await file.read()).decode("utf-8")
    templates_store[slot_id] = {
        "title": title,
        "filename": file.filename,
        "content": content,
    }
    return {"message": "Template uploaded successfully.", "filename": file.filename, "slot_id": slot_id, "title": title}


@app.get("/templates")
def list_templates():
    """Return all uploaded CV slots (without full content for lighter response)."""
    return {
        slot_id: {
            "title": data["title"],
            "filename": data["filename"],
            "exists": True,
        }
        for slot_id, data in templates_store.items()
    }


@app.delete("/templates/{slot_id}")
def delete_template(slot_id: str):
    if slot_id in templates_store:
        del templates_store[slot_id]
        return {"message": "Deleted"}
    raise HTTPException(status_code=404, detail="Slot not found.")


@app.post("/optimize")
def optimize_resume(req: OptimizeRequest):
    if req.slot_id not in templates_store:
        raise HTTPException(status_code=404, detail="No CV found for this slot. Please upload one first.")

    template_content = templates_store[req.slot_id]["content"]

    # Step 1: Extract keywords from JD
    extraction_prompt = f"""You are an ATS expert. Analyze the following job description and extract key information.

Return ONLY a valid JSON object. No explanation, no markdown fences, no extra text. Format exactly:
{{
  "technical_skills": [],
  "soft_skills": [],
  "keywords": [],
  "technologies": [],
  "experience_level": "",
  "education": "",
  "certifications": []
}}

Job Description:
{req.job_description}"""

    raw = chat(extraction_prompt, temperature=0.2)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        analysis = json.loads(raw)
    except Exception:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        try:
            analysis = json.loads(match.group()) if match else {"raw": raw}
        except Exception:
            analysis = {"raw": raw}

    # Step 2: Optimize resume
    optimization_prompt = f"""You are an expert ATS resume optimizer. Update the LaTeX CV below to be optimized for the given job and to PASS ATS keyword-matching scans — while staying on ONE PAGE.

STRICT RULES:
1. Keep ALL LaTeX structure, packages, and formatting commands exactly as-is.
2. From the Job Analysis below, identify which "technical_skills", "technologies", and "keywords" are MOST RELEVANT and HIGH-IMPACT for this specific role (the core requirements, not minor/generic mentions). Prioritize these. Add the relevant ones into the Skills section using the EXACT same wording/spelling as in the Job Analysis (ATS scans do exact string matches), under the most fitting category.
2b. REORDER the Skills section: place the skill categories and individual skills MOST RELEVANT to this job FIRST (left-to-right and top-to-bottom), with less relevant ones later. Within each category, list the most job-relevant skills first.
3. SKIP keywords that are generic, low-impact, or not core to the role — do not stuff irrelevant terms just to pad the list. Quality and relevance over quantity.
4. Naturally integrate the chosen keywords into bullet points too, rewriting bullet points to mirror the job's language/terminology where truthful.
4b. ANTI-REPETITION RULE — this is critical for readability and avoiding "keyword-stuffed" appearance to human recruiters: do NOT repeat the same phrase, term, or buzzword (e.g. a specific framework name, "design system", "modern tech stack", "troubleshooting", a specific methodology) more than ONCE or TWICE across the entire CV, and never in consecutive or nearby bullet points. Each bullet point must describe a DIFFERENT accomplishment or responsibility — vary the verbs, structure, and vocabulary naturally, the way a human would write about different experiences. If a keyword is already well-represented once in the Skills section, you generally do NOT need to repeat it again in multiple bullet points — one strong, natural mention is often enough.
4c. Write the way a skilled human resume writer would: each bullet should sound distinct and specific, not like variations of the same templated sentence. Avoid the pattern of stating the same skill/tool in nearly identical phrasing across multiple bullets or sections.
5. Do NOT invent NEW work experience, job titles, companies, or dates. You MAY add relevant skills/tools to the Skills section even if not explicitly used in past roles, as long as they don't contradict the person's background.
6. Do NOT add tables, graphics, or complex structures.
7. Do NOT add a Professional Summary or objective section under any circumstances.
8. The CV MUST fit on exactly ONE page — this is non-negotiable. If adding skills would overflow the page, drop the lowest-priority keywords first rather than overflow.
9. FINAL CHECK before output: scan your own draft for any term/phrase repeated 3+ times across the document (especially across multiple bullet points) and rewrite those occurrences with different wording or remove the redundant mention. The CV should read as natural and professional, not as if keywords were inserted everywhere for ATS purposes.
10. Return ONLY the complete updated LaTeX content. No explanation, no markdown fences, no ```latex.

Job Analysis:
{json.dumps(analysis, indent=2)}

Original LaTeX CV:
{template_content}"""

    optimized_tex = chat(optimization_prompt, temperature=0.4)

    if optimized_tex.startswith("```"):
        optimized_tex = re.sub(r"^```[a-z]*\n?", "", optimized_tex)
        optimized_tex = re.sub(r"\n?```$", "", optimized_tex)

    # Step 3: Check coverage of the most relevant keywords (informational only — no forced re-injection, to protect 1-page constraint)
    all_keywords = set()
    for field in ["technical_skills", "technologies", "keywords"]:
        for kw in analysis.get(field, []) if isinstance(analysis, dict) else []:
            if isinstance(kw, str) and kw.strip():
                all_keywords.add(kw.strip())

    tex_lower = optimized_tex.lower()
    missing = [kw for kw in all_keywords if kw.lower() not in tex_lower]
    covered = [kw for kw in all_keywords if kw.lower() in tex_lower]

    # Step 4: ATS score for the final optimized CV against the job description
    score_prompt = f"""You are an ATS (Applicant Tracking System) scoring engine. Compare the LaTeX CV below against the job description and give it an ATS match score from 0-100.

Consider: keyword/skill overlap, relevance of experience, use of job-related terminology, and overall alignment with the role's requirements.

Return ONLY a valid JSON object, no explanation, no markdown fences. Format exactly:
{{
  "ats_score": 87,
  "summary": "one short sentence explaining the score"
}}

Job Description:
{req.job_description}

LaTeX CV:
{optimized_tex}"""

    score_raw = chat(score_prompt, temperature=0.2)
    score_raw = re.sub(r"```json|```", "", score_raw).strip()
    try:
        score_data = json.loads(score_raw)
    except Exception:
        match = re.search(r'\{.*\}', score_raw, re.DOTALL)
        try:
            score_data = json.loads(match.group()) if match else {}
        except Exception:
            score_data = {}

    ats_score = score_data.get("ats_score")
    ats_summary = score_data.get("summary", "")

    return {
        "job_id": req.job_id,
        "slot_id": req.slot_id,
        "analysis": analysis,
        "tex_content": optimized_tex,
        "ats_score": ats_score,
        "ats_summary": ats_summary,
        "keyword_coverage": {
            "total_keywords": len(all_keywords),
            "covered": len(covered),
            "missing": missing,
        },
    }


@app.post("/suggest-cv")
def suggest_cv(req: SuggestRequest):
    if not templates_store:
        raise HTTPException(status_code=404, detail="No CVs uploaded yet. Please upload at least one CV in Step 1.")

    # Build a short profile for each CV (title + first ~600 chars of content)
    cv_profiles = []
    for slot_id, data in templates_store.items():
        snippet = data["content"][:800]
        cv_profiles.append({
            "slot_id": slot_id,
            "title": data["title"],
            "snippet": snippet
        })

    prompt = f"""You are an ATS career-matching expert. Given a job description and a list of CVs (with title and content snippet), determine how well each CV matches the job, AND identify what type/title of CV would be IDEALLY suited for this job (regardless of what's uploaded).

For each uploaded CV, return a match percentage (0-100) representing how well its skills/experience align with the job description's requirements.

Also determine the IDEAL CV title for this job — a short role-type label (e.g. "Frontend Developer", "Backend Developer", "Full Stack Developer", "Data Scientist", "DevOps Engineer", "Mobile Developer", "Cloud Engineer", etc.) based purely on the job description's required skills, independent of the uploaded CVs.

Then compare: if the best-matching uploaded CV has match_percent >= 60, recommend using that CV (best_slot_id). If even the best uploaded CV has match_percent < 60, set best_slot_id to null and clearly state in "reason" that none of the uploaded CVs are a strong fit, and that the user should consider creating/uploading a CV titled like the "ideal_cv_title".

Return ONLY a valid JSON object, no explanation, no markdown fences. Format exactly:
{{
  "matches": [
    {{"slot_id": "slot1", "title": "Data Analyst", "match_percent": 85}},
    {{"slot_id": "slot2", "title": "QA", "match_percent": 40}}
  ],
  "ideal_cv_title": "Frontend Developer",
  "best_slot_id": "slot1",
  "reason": "short reason explaining the recommendation"
}}

Job Description:
{req.job_description}

CVs:
{json.dumps(cv_profiles, indent=2)}"""

    raw = chat(prompt, temperature=0.2)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        result = json.loads(raw)
    except Exception:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        try:
            result = json.loads(match.group()) if match else {"raw": raw}
        except Exception:
            result = {"raw": raw}

    return result


@app.get("/health")
def health():
    return {"status": "ok"}