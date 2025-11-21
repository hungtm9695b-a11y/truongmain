import os
import base64
import json
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# PROMPTS
# ============================================================

VISION_PROMPT = """
Bạn là chuyên gia tim mạch theo ESC 2023.
Hãy đọc ECG chụp giấy và trả về:
- Nhịp
- Block
- Trục
- QRS, QTc
- ST chênh lên / chênh xuống
- Sóng T
- Q bệnh lý
- Gợi ý STEMI/NSTEMI
Kết luận 1–2 câu.
"""

CLINICAL_PROMPT = """
Triệu chứng theo ESC:
- Vị trí: {loc}
- Tính chất: {quality}
- Khởi phát: {trigger}
- Giảm đau: {relief}
- Kèm theo: {assoc}
- Diễn tiến: {dynamic}
- Không do tim: {noncardiac}

ESC criteria: {esc_criteria}

Phân loại:
- 3 tiêu chí → dien_hinh
- 2 tiêu chí → khong_dien_hinh
- 0–1 → it_goi_y

Chỉ trả về 1 từ: dien_hinh / khong_dien_hinh / it_goi_y.
"""

# JSON trong prompt phải escape {{ }}
FUSION_PROMPT = """
Bạn là chuyên gia ESC.

ECG:
{ecg_text}

Triệu chứng:
{symptom_type}

Hãy trả về JSON:
{{
 "muc_nguy_co": "...",
 "chan_doan_goi_y": "...",
 "khuyen_cao": ["...", "..."]
}}
"""


# ============================================================
# API
# ============================================================

@app.post("/api/analyze")
async def analyze(
    ecg_file: UploadFile,
    loc: str = Form("none"),
    quality: str = Form("none"),
    trigger: str = Form("none"),
    relief: str = Form("none"),
    assoc: str = Form("none"),
    dynamic: str = Form("none"),
    noncardiac: str = Form("none"),
    esc_criteria: str = Form("none"),
):
    # ------------------------------
    # 1) Vision: đọc ECG bằng GPT-4o
    # ------------------------------
    raw = await ecg_file.read()
    b64 = base64.b64encode(raw).decode()

    vision_payload = [
        {"role": "system", "content": VISION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Đọc ECG sau:"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
            ]
        }
    ]

    vision_res = client.responses.create(
        model="gpt-4o",
        input=vision_payload
    )

    ecg_text = vision_res.output_text.strip()

    # ------------------------------
    # 2) Triệu chứng ESC
    # ------------------------------
    clinical_prompt = CLINICAL_PROMPT.format(
        loc=loc, quality=quality, trigger=trigger,
        relief=relief, assoc=assoc, dynamic=dynamic,
        noncardiac=noncardiac, esc_criteria=esc_criteria
    )

    clinical_res = client.responses.create(
        model="gpt-4o-mini",
        input=clinical_prompt
    )

    symptom_type = clinical_res.output_text.strip()
    if symptom_type not in ["dien_hinh", "khong_dien_hinh", "it_goi_y"]:
        symptom_type = "it_goi_y"

    # ------------------------------
    # 3) Fusion ESC (risk + dx + rec)
    # ------------------------------
    fusion_prompt = FUSION_PROMPT.format(
        ecg_text=ecg_text,
        symptom_type=symptom_type
    )

    fusion_res = client.responses.create(
        model="gpt-4o-mini",
        input=fusion_prompt
    )

    fusion_json = json.loads(fusion_res.output_text)

    # ------------------------------
    # 4) Final output
    # ------------------------------
    return {
        "ecg": ecg_text,
        "phan_loai_trieu_chung": symptom_type,
        "muc_nguy_co": fusion_json["muc_nguy_co"],
        "chan_doan_goi_y": fusion_json["chan_doan_goi_y"],
        "khuyen_cao": fusion_json["khuyen_cao"]
    }


# Local run
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
