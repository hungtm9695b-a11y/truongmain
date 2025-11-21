# ============================
#  AI ECG BACKEND - FIXED & OPTIMIZED
#  Updated for OpenAI SDK v1.0+ & ESC Guidelines
# ============================

import os
import json
import base64
import uvicorn
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình Client (Nên dùng biến môi trường để bảo mật)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE"))

# ============================================================
# 1) PROMPTS (GIỮ NGUYÊN LOGIC CỦA BẠN)
# ============================================================

VISION_PROMPT = """
Bạn là chuyên gia tim mạch theo chuẩn ESC.
Nhiệm vụ: Đọc ảnh ECG và trả về kết quả dưới dạng văn bản ngắn gọn nhưng đầy đủ thông số kỹ thuật.

PHÂN TÍCH CHI TIẾT:
- Nhịp, Tần số, Trục.
- Phân tích sóng P, QRS (biên độ, thời gian), khoảng PR, QT.
- Đánh giá đoạn ST (chênh lên/xuống bao nhiêu mm, ở chuyển đạo nào).
- Đánh giá sóng T (dẹt, âm, cao nhọn).
- Tìm dấu hiệu nhồi máu cơ tim (STEMI/NSTEMI) hoặc tương đương STEMI (Sgarbossa, De Winter, Wellens).

KẾT LUẬN CẦN TRẢ VỀ:
Mô tả tóm tắt các bất thường tìm thấy. Nếu bình thường ghi "ECG trong giới hạn bình thường".
"""

CLINICAL_PROMPT = """
Bạn là chuyên gia tim mạch ESC. Dựa vào dữ liệu bệnh nhân dưới đây, hãy phân loại mức độ điển hình của cơn đau ngực.

Dữ liệu bệnh nhân:
- Tuổi: {age}, Giới: {sex}
- Sinh hiệu: HA {sbp}/{dbp}, Mạch {hr}, SpO2 {spo2}
- Triệu chứng: Vị trí {loc}, Tính chất {quality}, Khởi phát {trigger}, Giảm đau {relief}
- Kèm theo: {assoc}, Diễn tiến: {dynamic}
- Yếu tố không do tim: {noncardiac}
- HEAR Score: {hear_score} ({hear_level})
- ESC Criteria Input: {esc_criteria}

NHIỆM VỤ:
Phân loại cơn đau ngực vào 1 trong 4 nhóm sau:
1. "dien_hinh" (Đáp ứng đủ 3 tiêu chuẩn ESC)
2. "khong_dien_hinh" (Đáp ứng 2 tiêu chuẩn)
3. "it_goi_y" (0-1 tiêu chuẩn)
4. "khong_co_du_lieu" (Thiếu thông tin)

Chỉ trả về đúng 1 từ khóa trong 4 từ trên. Không giải thích thêm.
"""

FUSION_PROMPT = """
Bạn là chuyên gia cấp cứu tim mạch.

DỮ LIỆU ĐẦU VÀO:
1. Kết quả đọc ECG: 
{ecg_text}

2. Phân loại triệu chứng lâm sàng: 
{symptom_type}

NHIỆM VỤ:
Tổng hợp và trả về kết quả định dạng JSON (nghiêm ngặt).

YÊU CẦU OUTPUT JSON:
{{
  "muc_nguy_co": "cao" | "trung_binh" | "thap",
  "chan_doan_goi_y": "Câu chẩn đoán ngắn gọn",
  "khuyen_cao": [
      "Khuyến cáo 1 (Hành động ngay)",
      "Khuyến cáo 2 (Cận lâm sàng/Theo dõi)"
  ]
}}
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
    try:
        # ======================
        # BƯỚC 1: VISION (ĐỌC ẢNH)
        # ======================
        content = await ecg_file.read()
        b64_image = base64.b64encode(content).decode('utf-8')

        # Sửa lỗi 1: Dùng gpt-4o và đúng cấu trúc message Vision
        vision_res = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": VISION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hãy phân tích hình ảnh ECG này."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        ecg_text = vision_res.choices[0].message.content

        # ============================
        # BƯỚC 2: LÂM SÀNG (TEXT)
        # ============================
        clinical_formatted = CLINICAL_PROMPT.format(
            age=age, sex=sex, sbp=sbp, dbp=dbp, hr=hr, spo2=spo2,
            loc=loc, quality=quality, trigger=trigger, relief=relief,
            assoc=assoc, dynamic=dynamic, noncardiac=noncardiac,
            esc_criteria=esc_criteria, hear_score=hear_score, hear_level=hear_level
        )

        # Sửa lỗi 2: Dùng gpt-4o-mini cho nhanh
        clinical_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": clinical_formatted}],
            temperature=0.0
        )
        
        raw_symptom = clinical_res.choices[0].message.content.strip().replace('"', '').lower()
        valid_types = ["dien_hinh", "khong_dien_hinh", "it_goi_y", "khong_co_du_lieu"]
        symptom_type = raw_symptom if raw_symptom in valid_types else "khong_co_du_lieu"

        # ============================
        # BƯỚC 3: TỔNG HỢP (JSON)
        # ============================
        fusion_formatted = FUSION_PROMPT.format(
            ecg_text=ecg_text,
            symptom_type=symptom_type
        )

        # Sửa lỗi 3: Bật chế độ JSON Mode để không bị lỗi parse
        fusion_res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": fusion_formatted}],
            response_format={"type": "json_object"}, 
            temperature=0.1
        )

        fusion_json = json.loads(fusion_res.choices[0].message.content)

        # ============================
        # TRẢ KẾT QUẢ
        # ============================
        return {
            "phan_loai_trieu_chung": symptom_type,
            "ecg": {
                "ket_luan_ecg": ecg_text
            },
            "muc_nguy_co": fusion_json.get("muc_nguy_co", "unknown"),
            "chan_doan_goi_y": fusion_json.get("chan_doan_goi_y", "Chưa rõ"),
            "khuyen_cao": fusion_json.get("khuyen_cao", [])
        }

    except Exception as e:
        print(f"Lỗi Server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
