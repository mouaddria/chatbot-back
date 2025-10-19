import os
import json
import re
import base64
from io import BytesIO
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

# --- Load environment ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in environment variables.")

client = OpenAI(api_key=API_KEY)
DEFAULT_OAI_MODEL = "gpt-4o"

# --- FastAPI app ---
app = FastAPI(title="Car Analyzer API")

# CORS: allow only your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatbot-front-three.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Prompts ---
CAR_SYSTEM_PROMPT = (
    "You are an expert automotive appraiser for the Moroccan market.\n"
    "From one or more car photos AND a user-stated condition (new|used), "
    "identify the car and return STRICT JSON only.\n"
    "If unsure, use 'unknown' and include multiple hypotheses with confidences.\n"
    "If condition=='new', indicate if this model is still available new in Morocco "
    "as 'available' | 'discontinued' | 'unclear'.\n"
    "Estimate price in MAD as a range (min,max) and state assumptions."
)

CAR_JSON_INSTRUCTIONS = (
    "Return ONLY this JSON object:\n"
    "{\n"
    '  "condition": "new|used",\n'
    '  "detected": {\n'
    '    "make": "string|unknown",\n'
    '    "model": "string|unknown",\n'
    '    "generation": "string|unknown",\n'
    '    "year_range": [null,null],\n'
    '    "fuel": "petrol|diesel|hybrid|electric|unknown",\n'
    '    "transmission": "manual|automatic|cvt|unknown",\n'
    '    "body_type": "sedan|hatchback|suv|coupe|wagon|van|pickup|unknown",\n'
    '    "trim": "string|unknown",\n'
    '    "color": "string|unknown",\n'
    '    "top_matches": [\n'
    '      {"candidate":"make model generation", "confidence":0.0, "evidence":["visual cue 1","visual cue 2"]}\n'
    '    ]\n'
    "  },\n"
    '  "new_availability": "available|discontinued|unclear",\n'
    '  "price_estimate": {"currency":"MAD","min":0,"max":0,"confidence":0.0,"basis":"short note"},\n'
    '  "assumptions": ["bullet reason 1","bullet reason 2"]\n'
    "}"
)

# --- Helpers ---
def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def to_image_content(pils: List[Image.Image]):
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(im)}"}}
        for im in pils
    ]

def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    return {"raw_response": text}

# --- OpenAI call ---
def call_openai_car(prompt: str, images: List[Image.Image], condition: str = "used",
                    model: str = DEFAULT_OAI_MODEL, temperature: float = 0.0, max_tokens: int = 800):
    user_content = [{"type": "text", "text": f"{prompt}\n\nUser_condition: {condition}\n\n{CAR_JSON_INSTRUCTIONS}"}]
    user_content += to_image_content(images)
    
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CAR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return _safe_json_loads(resp.choices[0].message.content)

# --- Routes ---
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/analyze-car")
async def analyze_car(
    condition: str = Form(..., pattern="^(new|used)$"),
    prompt: str = Form("Identify this car and estimate a fair price in Morocco."),
    model: str = Form(DEFAULT_OAI_MODEL),
    files: List[UploadFile] = File(...),
):
    try:
        pils = [Image.open(BytesIO(await f.read())).convert("RGB") for f in files]
        result = call_openai_car(prompt=prompt, images=pils, condition=condition, model=model)
        result.setdefault("condition", condition)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
