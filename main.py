import os
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File
from openai import OpenAI

# Khởi tạo App
app = FastAPI(title="ECG Cardiologist AI (Vision)")

# Cấu hình Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# System Prompt (Vẫn giữ nguyên tiêu chuẩn y khoa)
SYSTEM_PROMPT_CARDIOLOGIST = """
## VAI TRÒ (ROLE)
Bạn là một Chuyên gia Tim mạch (Cardiologist) cấp cao với 20 năm kinh nghiệm.
## TIÊU CHUẨN
Tuân thủ ESC/ACC/AHA Guidelines. Đọc kỹ hình ảnh ECG được cung cấp.
## OUTPUT FORMAT
Trả về Markdown:
1. Phân tích chuyên sâu (Mô tả sóng P, QRS, ST, T, Nhịp, Trục).
2. Kết luận.
3. Mức độ cảnh báo: [MỨC XANH]/[MỨC VÀNG]/[MỨC ĐỎ].
4. Khuyến cáo hành động (3 ý, mỗi ý 2 câu chuẩn y khoa).
"""

@app.get("/")
def home():
    return {"status": "ECG Vision Service is Running."}

@app.post("/analyze-image")
async def analyze_ecg_image(file: UploadFile = File(...)):
    """
    Upload ảnh ECG (JPG/PNG) để AI phân tích.
    """
    if not client.api_key:
        raise HTTPException(status_code=500, detail="Chưa cấu hình OpenAI API Key")
    
    try:
        # 1. Đọc file ảnh từ người dùng gửi lên
        contents = await file.read()
        
        # 2. Mã hóa ảnh sang Base64 để gửi cho OpenAI
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # 3. Gửi yêu cầu đến GPT-4o (Vision capabilities)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": SYSTEM_PROMPT_CARDIOLOGIST
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hãy đọc hình ảnh điện tim này và chẩn đoán theo đúng quy trình đã học."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        return {"result": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")
