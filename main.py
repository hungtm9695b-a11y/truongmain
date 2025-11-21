import os
import base64
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# ==========================
# FastAPI + CORS
# ==========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# OpenAI Client
# ==========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================
# PROMPTS
# ==========================

ECG_PROMPT = """
Bạn là bác sĩ tim mạch chuyên sâu theo ESC 2024 + ACC/AHA 2021.
Phân tích ECG theo 7 bước chuẩn:
- Tần số
- Nhịp
- Trục
- PR–QRS–QTc
- Sóng P
- QRS: block, rộng/hẹp
- ST–T: ST chênh lên/giảm, T đảo, Q bệnh lý

Trọng tâm đánh giá thiếu máu cơ tim: STEMI theo vùng (trước, vách, bên, dưới), NSTEMI, thiếu máu cơ tim có thể.

Trả về JSON:
{
  "ecg_analysis": "...",
  "ischemia_risk": { "level": "Thấp/Trung bình/Cao", "reason": "..." },
  "final_ecg_conclusion": "1–2 câu."
}
"""

CLINICAL_PROMPT = """
Bạn là bác sĩ tim mạch theo ESC 2024.
Dựa vào triệu chứng:
1. Phân loại đau ngực: điển hình / không điển hình / không gợi ý thiếu máu cơ tim.
2. Phân tầng nguy cơ: thấp / trung bình / cao.
3. Kết hợp dữ liệu ECG đã cung cấp.

Trả về JSON:
{
  "clinical_summary": "...",
  "combined_risk_level": "...",
  "reason": "...",
  "recommendations": [
    { "title": "Khuyến cáo 1", "content": "2 câu" },
    { "title": "Khuyến cáo 2", "content": "2 câu" },
    { "title": "Khuyến cáo 3", "content": "2 câu" }
  ]
}
"""

FINAL_PROMPT = """
Tổng hợp dữ liệu:
ECG: {ECG_DATA}
Lâm sàng: {CLINICAL_DATA}

Nhiệm vụ:
- Phân tầng nguy cơ thiếu máu cơ tim (Thấp/Trung bình/Cao)
- Kết luận 1–2 câu
- 3 khuyến cáo đúng chuẩn ESC (mỗi khuyến cáo 2 câu)

Trả về JSON:
{
  "ecg": "...",
  "clinical": "...",
  "risk_level": "...",
  "conclusion": "...",
  "recommendations": [
    { "title": "Khuyến cáo 1", "content": "2 câu" },
    { "title": "Khuyến cáo 2", "content": "2 câu" },
    { "title": "Khuyến cáo 3", "content": "2 câu" }
  ]
}
"""

# ==========================
# API 1 — PHÂN TÍCH ECG
# ==========================
@app.post("/analyze-ecg")
async def analyze_ecg(ecg_file: UploadFile):

    image_bytes = await ecg_file.read()
    b64 = base64.b64encode(image_bytes).decode()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ECG_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64}"
                    }
                ]
            }
        ],
        temperature=0.1,
    )

    return response.choices[0].message["content"]


# ==========================
# API 2 — LÂM SÀNG + TỔNG HỢP
# ==========================
@app.post("/analyze-clinical")
async def analyze_clinical(
    ecg_data: str = Form(...),
    symptoms: str = Form(...)
):

    # --- Bước 1: phân tích lâm sàng ---
    clinical_step = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CLINICAL_PROMPT},
            {"role": "user", "content": symptoms},
            {"role": "user", "content": f"ECG: {ecg_data}"}
        ],
        temperature=0.1,
    )

    clinical_json = clinical_step.choices[0].message["content"]

    # --- Bước 2: tổng hợp ---
    final_prompt_filled = FINAL_PROMPT \
        .replace("{ECG_DATA}", ecg_data) \
        .replace("{CLINICAL_DATA}", clinical_json)

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": final_prompt_filled},
            {"role": "user", "content": "Tạo JSON cuối cùng."}
        ],
        temperature=0.1,
    )

    return final_response.choices[0].message["content"]


# ==========================
# TEST ROOT
# ==========================
@app.get("/")
def home():
    return {"message": "ECG AI API is running"}
