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
    optimization_prompt = f"""You are an expert ATS resume optimizer. Update the LaTeX CV below to be FULLY optimized for the given job and to PASS ATS keyword-matching scans.

STRICT RULES:
1. Keep ALL LaTeX structure, packages, and formatting commands exactly as-is.
2. CRITICAL — Every single item in "technical_skills", "technologies", and "keywords" from the Job Analysis below MUST appear somewhere in the CV, using the EXACT same wording/spelling as in the Job Analysis (ATS scans do exact string matches). If a skill is not currently in the CV, add it to the Skills section as a new entry under the most relevant category (or create a new category if needed).
3. Naturally integrate keywords into bullet points and rewrite bullet points to mirror the job's language/terminology where truthful.
4. Do NOT invent NEW work experience, job titles, companies, or dates. You MAY add additional skills/tools to the Skills section even if not explicitly used in past roles, as long as they don't contradict the person's background — treat the Skills section as a keyword-coverage section, not a claims section.
5. Do NOT add tables, graphics, or complex structures.
6. Do NOT add a Professional Summary or objective section under any circumstances.
7. The CV MUST fit on exactly ONE page. Use concise phrasing and tight skill lists to fit everything.
8. Return ONLY the complete updated LaTeX content. No explanation, no markdown fences, no ```latex.

Job Analysis (ALL of these keywords/skills/technologies MUST be present verbatim somewhere in the output):
{json.dumps(analysis, indent=2)}

Original LaTeX CV:
{template_content}"""

    optimized_tex = chat(optimization_prompt, temperature=0.4)

    if optimized_tex.startswith("```"):
        optimized_tex = re.sub(r"^```[a-z]*\n?", "", optimized_tex)
        optimized_tex = re.sub(r"\n?```$", "", optimized_tex)

    # Step 3: Verify keyword coverage — find any ATS keywords missing from the output
    all_keywords = set()
    for field in ["technical_skills", "technologies", "keywords"]:
        for kw in analysis.get(field, []) if isinstance(analysis, dict) else []:
            if isinstance(kw, str) and kw.strip():
                all_keywords.add(kw.strip())

    tex_lower = optimized_tex.lower()
    missing = [kw for kw in all_keywords if kw.lower() not in tex_lower]

    if missing:
        # Step 4: Force-inject missing keywords into the Skills section
        fix_prompt = f"""The following LaTeX CV is missing these ATS keywords (exact wording required for ATS matching):
{json.dumps(missing, indent=2)}

Add ALL of these missing keywords into the existing Skills section of the LaTeX CV below, using the exact wording given, placed under the most relevant category (or add to an existing category's list with commas). Do NOT remove or change anything else. Do NOT add a new section. Keep the CV fitting on ONE page — if needed, slightly shorten existing skill descriptions to make room.

Return ONLY the complete updated LaTeX content. No explanation, no markdown fences.

LaTeX CV:
{optimized_tex}"""

        fixed_tex = chat(fix_prompt, temperature=0.3)
        if fixed_tex.startswith("```"):
            fixed_tex = re.sub(r"^```[a-z]*\n?", "", fixed_tex)
            fixed_tex = re.sub(r"\n?```$", "", fixed_tex)

        # Re-check coverage after fix
        fixed_lower = fixed_tex.lower()
        still_missing = [kw for kw in missing if kw.lower() not in fixed_lower]
        if len(still_missing) < len(missing):
            optimized_tex = fixed_tex
            missing = still_missing

    return {
        "job_id": req.job_id,
        "slot_id": req.slot_id,
        "analysis": analysis,
        "tex_content": optimized_tex,
        "keyword_coverage": {
            "total_keywords": len(all_keywords),
            "covered": len(all_keywords) - len(missing),
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

    prompt = f"""You are an ATS career-matching expert. Given a job description and a list of CVs (with title and content snippet), determine how well each CV matches the job.

For each CV, return a match percentage (0-100) representing how well its skills/experience align with the job description's requirements. Also return the slot_id of the single best match, and a short 1-sentence reason.

Return ONLY a valid JSON object, no explanation, no markdown fences. Format exactly:
{{
  "matches": [
    {{"slot_id": "slot1", "title": "Data Analyst", "match_percent": 85}},
    {{"slot_id": "slot2", "title": "QA", "match_percent": 40}}
  ],
  "best_slot_id": "slot1",
  "reason": "short reason why this CV is the best fit"
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
