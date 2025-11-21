import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# Khởi tạo App
app = FastAPI(title="ECG Cardiologist AI")

# Cấu hình Client (Lấy Key từ biến môi trường trên Render)
# LƯU Ý: Không hard-code API Key ở đây để bảo mật
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# System Prompt (Giữ nguyên như cũ)
SYSTEM_PROMPT_CARDIOLOGIST = """
## VAI TRÒ (ROLE)
Bạn là một Chuyên gia Tim mạch (Cardiologist) cấp cao với 20 năm kinh nghiệm.
## TIÊU CHUẨN
Tuân thủ ESC/ACC/AHA Guidelines.
## OUTPUT FORMAT
Trả về Markdown:
1. Phân tích chuyên sâu.
2. Kết luận.
3. Mức độ cảnh báo: [MỨC XANH]/[MỨC VÀNG]/[MỨC ĐỎ].
4. Khuyến cáo hành động (3 ý, mỗi ý 2 câu chuẩn y khoa).
"""

# Định nghĩa dữ liệu đầu vào
class ECGRequest(BaseModel):
    description: str  # Ví dụ: "ST chênh xuống V5, V6..."

@app.get("/")
def home():
    return {"status": "ECG AI Service is Running. Go to /docs to test."}

@app.post("/analyze")
def analyze_ecg(data: ECGRequest):
    if not client.api_key:
        raise HTTPException(status_code=500, detail="Chưa cấu hình OpenAI API Key")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CARDIOLOGIST},
                {"role": "user", "content": f"Dữ liệu ECG: {data.description}"}
            ],
            temperature=0.1,
        )
        return {"result": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
