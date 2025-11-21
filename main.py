import os
import base64
import json
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn

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
# OpenAI client
# ==========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================
# PROMPTS
# ==========================

VISION_PROMPT = """
Bạn là chuyên gia tim mạch theo ESC 2023.
Đọc ECG chụp giấy và mô tả chi tiết:
- Nhịp
- Trục
- QRS, QTc
- Block
- ST chênh / ST giảm
- Sóng T
- Q bệnh lý
- Gợi ý STEMI / NSTEMI
Kết luận 1–2 câu.
"""

CLINICAL_PROMPT = """
Dựa vào triệu chứng ESC:

- Vị trí: {loc}
- Tính chất: {quality}
- Khởi phát: {trigger}
- Giảm đau: {relief}
- Kèm theo: {assoc}
- Diễn tiến: {dynamic}
- Không do tim: {noncardiac}

ESC:
- 3 tiêu chí → dien_hinh
- 2 tiêu chí → khong_dien_hinh
- 0–1 → it_goi_y

Chỉ trả về duy nhất 1 từ:
- dien_hinh
- khong_dien_hinh
- it_goi_y
"""

# Escape JSON bằng {{ }}
FUSION_PROMPT = """
Bạn là chuyên gia tim mạch ESC 2023.

ECG phân tích:
{ecg_text}

Triệu chứng:
{symptom_type}

Hãy đánh giá ESC:
- Nguy cơ: thấp / trung bình / cao
- Chẩn đoán gợi ý
- 2 khuyến cáo ngắn gọn, súc tích, chuẩn ESC.

Trả về JSON DUY NHẤT:
{{
 "muc_nguy_co": "",
 "chan_doan_goi_y": "",
 "khuyen_cao": ["", ""]
}}
"""

# ==========================
# API
# ==========================
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
):

    # ==========================
    # 1) Vision: đọc ECG (gpt-4o)
    # ==========================
    raw = await ecg_file.read()
    b64 = base64.b64encode(raw).decode()

    vision_input = [
        {"role": "system", "content": VISION_PROMPT},
        {"role": "user", "content": [
            {"type": "input_text", "text": "Đọc ECG sau:"},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
        ]}
    ]

    vision_res = client.responses.create(
        model="gpt-4o",
        input=vision_input,
    )

    ecg_text = vision_res.output_text.strip()

    # ==========================
    # 2) Clinical symptoms
    # ==========================
    clinical_prompt = CLINICAL_PROMPT.format(
        loc=loc, quality=quality, trigger=trigger,
        relief=relief, assoc=assoc, dynamic=dynamic, noncardiac=noncardiac
    )

    clinical_res = client.responses.create(
        model="gpt-4o-mini",
        input=clinical_prompt
    )

    symptom_type = clinical_res.output_text.strip()
    if symptom_type not in ["dien_hinh", "khong_dien_hinh", "it_goi_y"]:
        symptom_type = "it_goi_y"

    # ==========================
    # 3) Fusion ESC (ép JSON)
    # ==========================
    fusion_prompt = FUSION_PROMPT.format(
        ecg_text=ecg_text,
        symptom_type=symptom_type
    )

    fusion_res = client.responses.create(
        model="gpt-4o",                 # BẮT BUỘC dùng gpt-4o để đảm bảo JSON
        input=fusion_prompt,
        response_format={"type": "json_object"}   # ÉP JSON 100%
    )

    fusion_json = json.loads(fusion_res.output_text)

    # ==========================
    # 4) Final output
    # ==========================
    return {
        "ecg": ecg_text,
        "phan_loai_trieu_chung": symptom_type,
        "muc_nguy_co": fusion_json.get("muc_nguy_co", ""),
        "chan_doan_goi_y": fusion_json.get("chan_doan_goi_y", ""),
        "khuyen_cao": fusion_json.get("khuyen_cao", [])
    }


# ==========================
# LOCAL RUN
# ==========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
