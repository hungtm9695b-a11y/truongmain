# ============================
#  AI ECG BACKEND - FULL main.py (FINAL VERSION)
#  By ChatGPT - optimized for ESC 2023 acute chest pain workflow
# ============================

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import json
from openai import OpenAI

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key="YOUR_API_KEY")

# ============================================================
# 1) VISION PROMPT – ĐỌC ECG NHƯ CHUYÊN GIA TIM MẠCH ESC
# ============================================================

VISION_PROMPT = """
Bạn là chuyên gia tim mạch theo chuẩn ESC 2023.
Hãy đọc ảnh ECG (dạng chụp giấy) với độ chính xác cao nhất:

PHÂN TÍCH CHI TIẾT (nhưng diễn giải gọn):
- Nhịp: xoang / nhanh / chậm / ngoại tâm thu / rung nhĩ / cuồng nhĩ / block AV
- Block dẫn truyền: RBBB, LBBB, hemiblock
- Trục điện tim (axis)
- QRS, QTc (ước tính)
- ST segment từng đạo trình:
  + ST chênh lên (số mm + đạo trình + vùng)
  + ST chênh xuống (mm + đạo trình)
  + Sóng T âm / sâu / đối xứng
  + Q bệnh lý
- Nhận diện STEMI theo vùng:
  + Trước, trước bên, bên, dưới, sau, thất phải
- Nhận diện “STEMI tương đương”:
  + ST chênh xuống V1–V3 gợi ý thành sau
  + T đảo sâu đối xứng trong thiếu máu nặng
  + LBBB mới + tiêu chuẩn Sgarbossa
  + RBBB kèm ST thay đổi

XUẤT KẾT LUẬN NGẮN GỌN:
- Nhịp
- Bất thường ST–T
- Có/Không STEMI hoặc ACS
- Vị trí tổn thương nếu có
"""


# ============================================================
# 2) CLINICAL PROMPT – PHÂN LOẠI TRIỆU CHỨNG THEO ESC
# ============================================================

CLINICAL_PROMPT = """
Bạn là chuyên gia tim mạch theo ESC 2023.

Dữ liệu bệnh nhân:
Tuổi: {age}
Giới: {sex}
SBP: {sbp}
DBP: {dbp}
Mạch: {hr}
SpO2: {spo2}

TRIỆU CHỨNG (checkbox):
- Vị trí: {loc}
- Tính chất: {quality}
- Khởi phát: {trigger}
- Giảm đau: {relief}
- Triệu chứng kèm: {assoc}
- Diễn tiến: {dynamic}
- Không do tim: {noncardiac}

HEAR Score (tham khảo): {hear_score} – mức {hear_level}

ESC CRITERIA: {esc_criteria}

PHÂN LOẠI CHUẨN ESC:
- 3 tiêu chí → "dien_hinh"
- 2 tiêu chí → "khong_dien_hinh"
- 0–1 tiêu chí → "it_goi_y"
- Thiếu dữ liệu → "khong_co_du_lieu"

Chỉ trả về duy nhất một trong bốn:
"dien_hinh", "khong_dien_hinh", "it_goi_y", "khong_co_du_lieu"
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

2) Chẩn đoán gợi ý (1 câu, cụ thể nhưng gọn):
- Nguy cơ cao: “ACS nguy cơ cao, phù hợp STEMI/NSTEMI; cần xử trí khẩn.”
- Trung bình: “Nghi ACS nguy cơ trung bình; cần theo dõi ECG + troponin động học.”
- Thấp: “Đau ngực nguy cơ thấp; khả năng ACS thấp.”

3) Khuyến cáo (đúng 2 câu, cụ thể nhưng ngắn):
- Nguy cơ cao:
  1. “Chuyển ngay cơ sở có PCI 24/7 và duy trì monitoring.”
  2. “Ưu tiên PCI khẩn; không trì hoãn điều trị.”
- Trung bình:
  1. “Theo dõi ECG, huyết động và troponin động học 0–1h.”
  2. “Nhập viện nếu triệu chứng còn hoặc troponin tăng.”
- Thấp:
  1. “Có thể xuất viện an toàn kèm hướng dẫn theo dõi.”
  2. “Quay lại ngay nếu đau ngực tái phát hoặc có dấu hiệu cảnh báo.”

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
    hear_score: str = Form("none"),
    hear_level: str = Form("none")
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
                {"type": "input_text", "text": "Phân tích ECG sau:"},
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
        age=age, sex=sex, sbp=sbp, dbp=dbp,
        hr=hr, spo2=spo2,
        loc=loc, quality=quality, trigger=trigger,
        relief=relief, assoc=assoc, dynamic=dynamic,
        noncardiac=noncardiac,
        esc_criteria=esc_criteria,
        hear_score=hear_score, hear_level=hear_level
    )

    clinical_res = client.responses.create(
        model="gpt-4.1-mini",
        input=clinical_prompt
    )

    symptom_type = clinical_res.output_text.strip()
    if symptom_type not in ["dien_hinh", "khong_dien_hinh", "it_goi_y", "khong_co_du_lieu"]:
        symptom_type = "khong_co_du_lieu"

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
# RUN BACKEND
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
