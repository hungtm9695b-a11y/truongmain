import os
import base64
import imghdr
import json
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import uvicorn

# ==========================
# FASTAPI + CORS
# ==========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# OPENAI CLIENT
# ==========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================
# PROMPTS
# ==========================
VISION_PROMPT = """
Bạn là chuyên gia tim mạch theo ESC 2023.
Hãy phân tích ECG theo 7 bước:
- Tần số
- Rhythm
- Trục
- PR – QRS – QTc
- Sóng P
- Phức bộ QRS
- ST – T (tìm ST chênh, ST giảm, T đảo)
- Gợi ý STEMI hay NSTEMI
Hãy trả về mô tả ngắn gọn, rõ ràng.
Kết luận cuối cùng đặt trong mục: 'ket_luan_ecg'.
"""

CLINICAL_PROMPT = """
Dựa vào triệu chứng ESC, phân loại:
- dien_hinh
- khong_dien_hinh
- it_goi_y

Triệu chứng:
- Vị trí: {loc}
- Tính chất: {quality}
- Khởi phát: {trigger}
- Giảm đau: {relief}
- Kèm theo: {assoc}
- Diễn tiến: {dynamic}
- Không do tim: {noncardiac}

Chỉ trả về duy nhất 1 từ.
"""

FUSION_PROMPT = """
Bạn là chuyên gia tim mạch ESC 2023.

ECG:
{ecg_text}

Triệu chứng ESC:
{symptom_type}

Dựa trên ESC 2023, hãy phân loại:
- muc_nguy_co: thap / trung_binh / cao
- chan_doan_goi_y
- 2 khuyen_cao cho tuyến cơ sở

Trả về đúng JSON:
{
 "muc_nguy_co": "",
 "chan_doan_goi_y": "",
 "khuyen_cao": ["", ""]
}
"""

# ==========================
# API CHÍNH
# ==========================
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
    risk: str = Form("none"),
    esc_criteria: str = Form("0"),
    hear_score: str = Form("0"),
    hear_level: str = Form("low"),
):

    # ==========================
    # 1) Vision: đọc ECG
    # ==========================
    raw = await ecg_file.read()
    b64 = base64.b64encode(raw).decode()
    img_type = imghdr.what(None, raw) or "jpeg"

    # ChatCompletion multimodal format
    messages_vision = [
        {"role": "system", "content": VISION_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Đọc ECG sau:"},
            {"type": "image_url", "image_url": f"data:image/{img_type};base64,{b64}"}
        ]}
    ]

    ecg_res = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_vision
    )

    ecg_text = ecg_res.choices[0].message.content.strip()

    # ==========================
    # 2) Phân loại triệu chứng ESC
    # ==========================
    clinical_prompt = CLINICAL_PROMPT.format(
        loc=loc,
        quality=quality,
        trigger=trigger,
        relief=relief,
        assoc=assoc,
        dynamic=dynamic,
        noncardiac=noncardiac
    )

    clinical_res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": clinical_prompt}
        ]
    )

    symptom_type = clinical_res.choices[0].message.content.strip()
    if symptom_type not in ["dien_hinh", "khong_dien_hinh", "it_goi_y"]:
        symptom_type = "it_goi_y"

    # ==========================
    # 3) Fusion JSON
    # ==========================
    fusion_prompt = FUSION_PROMPT.format(
        ecg_text=ecg_text,
        symptom_type=symptom_type
    )

    fusion_res = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": fusion_prompt}
        ]
    )

    fusion_json = json.loads(fusion_res.choices[0].message.content)

    # ==========================
    # 4) Chuẩn hóa output theo đúng HTML
    # ==========================
    return {
        "ecg": {
            "ket_luan_ecg": ecg_text
        },
        "phan_loai_trieu_chung": symptom_type,
        "muc_nguy_co": fusion_json.get("muc_nguy_co", "thap"),
        "chan_doan_goi_y": fusion_json.get("chan_doan_goi_y", ""),
        "khuyen_cao": fusion_json.get("khuyen_cao", ["", ""])
    }


# ==========================
# LOCAL RUN
# ==========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
