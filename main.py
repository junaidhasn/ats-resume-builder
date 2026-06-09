"""
ATS Resume Builder Agent - FastAPI Backend (Groq)
Deployment-ready version — no local file system dependency
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
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

# In-memory storage — no disk needed
template_store = {"content": None, "filename": None}


# ─── Models ───────────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    job_description: str
    job_id: str = "job-1"


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
async def upload_template(file: UploadFile = File(...)):
    if not file.filename.endswith(".tex"):
        raise HTTPException(status_code=400, detail="Only .tex files are accepted.")
    content = (await file.read()).decode("utf-8")
    template_store["content"] = content
    template_store["filename"] = file.filename
    return {"message": "Template uploaded successfully.", "filename": file.filename}


@app.get("/template-status")
def template_status():
    if template_store["content"]:
        return {"exists": True, "filename": template_store["filename"]}
    return {"exists": False}


@app.post("/optimize")
def optimize_resume(req: OptimizeRequest):
    if not template_store["content"]:
        raise HTTPException(status_code=404, detail="No CV template found. Please upload one first.")

    template_content = template_store["content"]

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
7. Return ONLY the complete updated LaTeX content. No explanation, no markdown fences, no ```latex.

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
        "analysis": analysis,
        "tex_content": optimized_tex,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
