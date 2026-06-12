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
    optimization_prompt = f"""You are an expert ATS resume optimizer. Update the LaTeX CV below to be optimized for the given job.

STRICT RULES:
1. Keep ALL LaTeX structure, packages, and formatting commands exactly as-is.
2. Naturally integrate the keywords and skills into skills section and bullet points ONLY.
3. Rewrite bullet points to match the job's language.
4. Do NOT add tables, graphics, or complex structures.
5. Do NOT invent experience or credentials. Only rephrase what already exists.
6. Do NOT add a Professional Summary or objective section under any circumstances.
7. The CV MUST fit on exactly ONE page. Do not add extra sections.
8. Return ONLY the complete updated LaTeX content. No explanation, no markdown fences, no ```latex.

Job Analysis:
{json.dumps(analysis, indent=2)}

Original LaTeX CV:
{template_content}"""

    optimized_tex = chat(optimization_prompt, temperature=0.4)

    if optimized_tex.startswith("```"):
        optimized_tex = re.sub(r"^```[a-z]*\n?", "", optimized_tex)
        optimized_tex = re.sub(r"\n?```$", "", optimized_tex)

    return {
        "job_id": req.job_id,
        "slot_id": req.slot_id,
        "analysis": analysis,
        "tex_content": optimized_tex,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
