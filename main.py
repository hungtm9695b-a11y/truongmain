# ============================================================
#  AI ECG BACKEND - FINAL VERSION FOR RENDER (GPT-4o VISION)
# ============================================================

import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import json
from openai import OpenAI

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client – API key từ Environment (Render)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# PROMPTS
# ============================================================

VISION_PROMPT = """
Bạn là chuyên gia tim mạch theo chuẩn ESC 2023.
Hãy đọc ECG chụp giấy và trả về phân tích chính xác:

- Nhịp
- Block nhánh
- Trục điện tim
- QRS, QTc (ước tính)
- ST chênh lên / xuống (mm + đạo trình)
- Sóng T bất thường
- Q bệnh lý
- Gợi ý STEMI theo vùng
- STEMI tương đương (Wellens, posterior MI…)

Kết luận cuối: 1–2 câu, ngắn gọn, rõ ràng.
"""

CLINICAL_PROMPT = """
Bạn là chuyên gia tim mạch theo ESC 2023.

Triệu chứng:
- Vị trí: {loc}
- Tính chất: {quality}
- Khởi phát: {trigger}
- Giảm đau: {relief}
- Kèm theo: {assoc}
- Diễn tiến: {dynamic}
- Không do tim: {noncardiac}

ESC criteria: {esc_criteria}

Quy tắc phân loại:
- 3 tiêu chí → "dien_hinh"
- 2 tiêu chí → "khong_dien_hinh"
- 0–1 tiêu chí → "it_goi_y"

Chỉ trả về đúng 1 từ.
"""

FUSION_PROMPT = """
Bạn là chuyên gia cấp cứu tim mạch ESC 2023.

ECG:
{ecg_text}

Triệu chứng ESC:
{symptom_type}

YÊU CẦU:
1) Phân loại nguy cơ: "cao", "trung_binh", "thap"
2) Chẩn đoán gợi ý: 1 câu
3) Khuyến cáo (2 câu):

- Nguy cơ cao:
  1. “Chuyển ngay cơ sở PCI 24/7.”
  2. “Không trì hoãn tái thông mạch.”

- Nguy cơ trung bình:
  1. “Theo dõi ECG + troponin động học.”
  2. “Nhập viện nếu triệu chứng còn.”

- Nguy cơ thấp:
  1. “Theo dõi ngoại trú.”
  2. “Khám lại nếu đau tăng hoặc thay đổi tính chất.”

Trả về JSON:
{
 "muc_nguy_co": "...",
 "chan_doan_goi_y": "...",
 "khuyen_cao": ["...", "..."]
}
"""

# ============================================================
# API /api/analyze
# ============================================================

@app.post("/api/analyze")
async def analyze(
    ecg_file: UploadFile,
    age: str = Form("none"),
    sex: str = Form("none"),
    sbp: str = Form("none"),
    dbp: str = Form("none"),
    hr: str = Form("none"),
    spo2: str = Form("none"),
    loc: str = Form("none"),
    quality: str = Form("none"),
    trigger: str = Form("none"),
    relief: str = Form("none"),
    assoc: str = Form("none"),
    dynamic: str = Form("none"),
    noncardiac: str = Form("none"),
    esc_criteria: str = Form("none"),
):

    # ======================================================
    # 1) Vision ECG — GPT-4o
    # ======================================================
    content = await ecg_file.read()
    b64 = base64.b64encode(content).decode()

    vision_input = [
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
        input=vision_input
    )
    ecg_text = vision_res.output_text

    # ======================================================
    # 2) Triệu chứng ESC
    # ======================================================
    clinical_prompt = CLINICAL_PROMPT.format(
        loc=loc, quality=quality, trigger=trigger,
        relief=relief, assoc=assoc, dynamic=dynamic,
        noncardiac=noncardiac, esc_criteria=esc_criteria
    )

    clin_res = client.responses.create(
        model="gpt-4o-mini",
        input=clinical_prompt
    )

    symptom_type = clin_res.output_text.strip()
    if symptom_type not in ["dien_hinh", "khong_dien_hinh", "it_goi_y"]:
        symptom_type = "it_goi_y"

    # ======================================================
    # 3) Fusion ESC (Risk + Diagnosis + Recommendation)
    # ======================================================
    fusion_prompt = FUSION_PROMPT.format(
        ecg_text=ecg_text,
        symptom_type=symptom_type
    )

    fusion_res = client.responses.create(
        model="gpt-4o-mini",
        input=fusion_prompt
    )

    fusion_json = json.loads(fusion_res.output_text)

    # ======================================================
    # 4) OUTPUT JSON
    # ======================================================

    return {
        "phan_loai_trieu_chung": symptom_type,
        "ecg": {"ket_luan_ecg": ecg_text},
        "muc_nguy_co": fusion_json["muc_nguy_co"],
        "chan_doan_goi_y": fusion_json["chan_doan_goi_y"],
        "khuyen_cao": fusion_json["khuyen_cao"]
    }


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
