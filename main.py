# ============================================================
#  AI ECG BACKEND - FULL main.py (FINAL VERSION FOR RENDER)
#  Lấy API key từ biến môi trường OPENAI_API_KEY
# ============================================================

import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import json
from openai import OpenAI

app = FastAPI()

# CORS cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# lấy API key từ biến môi trường (Render yêu cầu CÁCH NÀY)
client = OpenAI(api_key=os.getenv("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"))

# ============================================================
# 1) VISION PROMPT – ĐỌC ECG CHUẨN ESC
# ============================================================

VISION_PROMPT = """
Bạn là chuyên gia tim mạch theo chuẩn ESC 2023.
Hãy đọc ảnh ECG (dạng chụp giấy) với độ chính xác cao nhất:

PHÂN TÍCH CHI TIẾT:
- Nhịp: xoang / nhanh / chậm / ngoại tâm thu / rung nhĩ / cuồng nhĩ / block AV
- Block: RBBB, LBBB, hemiblock
- Trục điện tim
- QRS, QTc (ước tính)
- ST segment:
  + ST chênh lên (mm + đạo trình + vùng)
  + ST chênh xuống (mm + đạo trình)
  + Sóng T âm / sâu / đối xứng
  + Q bệnh lý
- Nhận diện STEMI theo vùng
- Nhận diện STEMI tương đương (posterior MI, Sgarbossa, Wellens…)

KẾT LUẬN NGẮN GỌN (1–2 câu):
- Nhịp
- Bất thường ST–T
- Có/Không dấu STEMI/NSTEMI
- Vị trí tổn thương nếu có
"""

# ============================================================
# 2) CLINICAL PROMPT ESC
# ============================================================

CLINICAL_PROMPT = """
Bạn là chuyên gia tim mạch theo ESC 2023.

TRIỆU CHỨNG theo ESC:
- Vị trí: {loc}
- Tính chất: {quality}
- Khởi phát: {trigger}
- Giảm đau: {relief}
- Triệu chứng kèm: {assoc}
- Diễn tiến: {dynamic}
- Không do tim: {noncardiac}

ESC CRITERIA (0–3): {esc_criteria}

PHÂN LOẠI:
- 3 tiêu chí → "dien_hinh"
- 2 tiêu chí → "khong_dien_hinh"
- 0–1 tiêu chí → "it_goi_y"

Chỉ trả về duy nhất một từ:
"dien_hinh", "khong_dien_hinh", "it_goi_y"
"""

# ============================================================
# 3) FUSION PROMPT – NGUY CƠ + CHẨN ĐOÁN + KHUYẾN CÁO
# ============================================================

FUSION_PROMPT = """
Bạn là chuyên gia cấp cứu tim mạch ESC 2023.

ECG:
{ecg_text}

Triệu chứng ESC:
{symptom_type}

NHIỆM VỤ:
1) Phân loại nguy cơ:
- "cao"
- "trung_binh"
- "thap"

2) Chẩn đoán gợi ý: 1 câu ngắn, chuẩn ESC

3) Khuyến cáo (2 câu):
- Nguy cơ cao:
  1. “Chuyển ngay cơ sở PCI 24/7.”
  2. “Không trì hoãn tái thông mạch.”
- Trung bình:
  1. “Theo dõi ECG + troponin động học.”
  2. “Nhập viện nếu triệu chứng còn.”
- Thấp:
  1. “Có thể theo dõi ngoại trú.”
  2. “Quay lại ngay nếu đau tăng.”

TRẢ VỀ JSON:
{
  "muc_nguy_co": "...",
  "chan_doan_goi_y": "...",
  "khuyen_cao": ["...", "..."]
}
"""

# ============================================================
# BACKEND API
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

    # ======================
    # 1) ĐỌC ECG VISION
    # ======================
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
        model="gpt-4.1-vision-preview",
        input=vision_input
    )
    ecg_text = vision_res.output_text

    # ============================
    # 2) TRIỆU CHỨNG ESC
    # ============================
    clinical_prompt = CLINICAL_PROMPT.format(
        loc=loc, quality=quality, trigger=trigger,
        relief=relief, assoc=assoc, dynamic=dynamic,
        noncardiac=noncardiac, esc_criteria=esc_criteria
    )

    clinical_res = client.responses.create(
        model="gpt-4.1-mini",
        input=clinical_prompt
    )

    symptom_type = clinical_res.output_text.strip()
    if symptom_type not in ["dien_hinh", "khong_dien_hinh", "it_goi_y"]:
        symptom_type = "it_goi_y"

    # ============================
    # 3) FUSION ESC
    # ============================
    fusion_prompt = FUSION_PROMPT.format(
        ecg_text=ecg_text,
        symptom_type=symptom_type
    )

    fusion_res = client.responses.create(
        model="gpt-4.1-mini",
        input=fusion_prompt
    )

    fusion_json = json.loads(fusion_res.output_text)

    # ============================
    # 4) JSON OUTPUT
    # ============================

    return {
        "phan_loai_trieu_chung": symptom_type,
        "ecg": {
            "ket_luan_ecg": ecg_text
        },
        "muc_nguy_co": fusion_json["muc_nguy_co"],
        "chan_doan_goi_y": fusion_json["chan_doan_goi_y"],
        "khuyen_cao": fusion_json["khuyen_cao"]
    }

# ============================================================
# RUN LOCAL
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
