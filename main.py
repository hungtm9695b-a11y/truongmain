import os
import base64
import json
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
Gợi ý STEMI / NSTEMI.
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

FUSION_PROMPT = """
Bạn là chuyên gia tim mạch ESC 2023.

ECG phân tích:
{ecg_text}

Triệu chứng:
{symptom_type}

Hãy đánh giá ESC:
- Nguy cơ: thấp / trung bình / cao
- Chẩn đoán gợi ý
- 2 khuyến cáo ngắn gọn chuẩn ESC.

Trả về JSON DUY NHẤT:
{{
 "muc_nguy_co": "",
 "chan_doan_goi_y": "",
 "khuyen_cao": ["", ""]
}}
"""


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
    # 1) Vision: đọc ECG
    # ==========================
    raw = await ecg_file.read()
    b64 = base64.b64encode(raw).decode()

    vision_input = [
        {"role": "system", "content": VISION_PROMPT},
        {"role": "user", "content": [
            {"type": "input_text", "text": "Đọc ECG sau:"},
            {
                "type": "input_image",
                "image": {"base64": b64}       # **** FIX LỖI QUAN TRỌNG ****
            }
        ]}
    ]

    vision_res = client.responses.create(
        model="gpt-4o",
        input=vision_input,
    )

    # Đọc text đúng format SDK mới
    ecg_text = vision_res.output[0].content[0].text.strip()

    # ==========================
    # 2) Triệu chứng
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
    # 3) Fusion JSON ESC
    # ==========================
    fusion_prompt = FUSION_PROMPT.format(
        ecg_text=ecg_text,
        symptom_type=symptom_type
    )

    fusion_res = client.responses.create(
        model="gpt-4o",
        input=fusion_prompt,
        response_format={"type": "json_object"}
    )

    try:
        fusion_json = json.loads(fusion_res.output_text)
    except:
        fusion_json = {
            "muc_nguy_co": "không xác định",
            "chan_doan_goi_y": "",
            "khuyen_cao": ["", ""]
        }

    return {
        "ecg": ecg_text,
        "phan_loai_trieu_chung": symptom_type,
        "muc_nguy_co": fusion_json.get("muc_nguy_co", ""),
        "chan_doan_goi_y": fusion_json.get("chan_doan_goi_y", ""),
        "khuyen_cao": fusion_json.get("khuyen_cao", []),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
