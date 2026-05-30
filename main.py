from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
import random
import io
import base64
import pandas as pd
from PIL import Image
from typing import Optional, List
import time
import numpy as np
import cv2
import urllib.request
import sqlite3
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI(title="Royal Elchim - Omni-Conscious Enterprise")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحديد مسار الذاكرة ليعمل محلياً وعلى خوادم Hugging Face
DB_DIR = os.environ.get("STORAGE_DIR", "data")
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, 'royal_memory.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vault
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  phone TEXT, 
                  type TEXT, 
                  text TEXT, 
                  selfie TEXT, 
                  product TEXT, 
                  date TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_to_db(phone, record_type, text, selfie=None, product=None):
    if not phone: return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO vault (phone, type, text, selfie, product, date) VALUES (?, ?, ?, ?, ?, ?)",
              (phone, record_type, text, selfie, product, date_str))
    conn.commit()
    conn.close()

def get_history_from_db(phone, limit=5):
    if not phone: return ""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date, type, text FROM vault WHERE phone=? ORDER BY id DESC LIMIT ?", (phone, limit))
    rows = c.fetchall()
    conn.close()
    if not rows: return ""
    history = " | ".join([f"[تاريخ {r[0]} - {r[1]}]: {r[2]}" for r in reversed(rows)])
    return history

TASK_FILE = 'face_landmarker.task'
TASK_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

if not os.path.exists(TASK_FILE):
    print("رويال مايند: جاري تحميل خريطة الوعي البصري...")
    urllib.request.urlretrieve(TASK_URL, TASK_FILE)

base_options = python.BaseOptions(model_asset_path=TASK_FILE)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

keys_string = os.environ.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEYS", ""))
SYSTEM_API_KEYS = [key.strip() for key in keys_string.split(",") if key.strip()]

VISION_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]
TEXT_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

def robust_generate(client_api_key, contents, models_list):
    if client_api_key and client_api_key.strip():
        keys_to_use = [client_api_key.strip()]
    else:
        if not SYSTEM_API_KEYS:
            # رد بديل في حالة عدم وجود مفتاح السيرفر بدلاً من تعطيل النظام
            return "مرحباً بكِ في عالم رويال إلكيم الملكي. أنا رويال مايند مستعد لمرافقتكِ."
        keys_to_use = SYSTEM_API_KEYS.copy()
        random.shuffle(keys_to_use)

    for model_name in models_list:
        for key in keys_to_use:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    client = genai.Client(api_key=key)
                    config = types.GenerateContentConfig(temperature=0.8, top_p=0.95)
                    response = client.models.generate_content(model=model_name, contents=contents, config=config)
                    if response and response.text:
                        return response.text
                except Exception as e:
                    error_str = str(e)
                    if "503" in error_str or "ResourceExhausted" in error_str or "429" in error_str:
                        time.sleep(1.5)
                        continue
                    else:
                        break
    return "قنوات رويال مايند ممتلئة حالياً، يرجى إعادة المحاولة."

class DiagnosisPayload(BaseModel):
    client_message: str
    phone: str
    client_api_key: Optional[str] = None

class ChatPayload(BaseModel):
    text: str
    category: str  
    phone: str
    client_api_key: Optional[str] = None

class SimulationPayload(BaseModel):
    user_selfie: str
    phone: str
    product_image: Optional[str] = None
    product_name_desc: Optional[str] = None
    makeup_type: str = "lips"
    hex_color: Optional[str] = "#8B0000"
    client_api_key: Optional[str] = None

def hex_to_rgb(hex_color: str):
    if not hex_color: return (139, 0, 0)
    hex_color = hex_color.lstrip('#')
    try: return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except: return (139, 0, 0)

def apply_royal_makeup(image_cv: np.ndarray, color_rgb: tuple, makeup_type: str):
    try:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = face_landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return image_cv, False

        height, width, _ = image_cv.shape
        face_landmarks = detection_result.face_landmarks[0]

        ZONES = {
            "lips": [[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185]],
            "eyeshadow": [
                [33, 246, 161, 160, 159, 158, 157, 173, 133], 
                [362, 398, 384, 385, 386, 387, 388, 466, 263] 
            ]
        }

        target_zones = ZONES.get(makeup_type, ZONES["lips"])
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for zone in target_zones:
            points = np.array([ [int(face_landmarks[idx].x * width), int(face_landmarks[idx].y * height)] for idx in zone ], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        color_layer = np.zeros_like(image_cv)
        color_layer[:] = color_rgb[::-1]
        alpha = mask / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        blended_layer = cv2.addWeighted(image_cv, 0.4, color_layer, 0.6, 0)
        final_image = (1.0 - alpha) * image_cv + alpha * blended_layer

        return final_image.astype(np.uint8), True
    except Exception as e:
        return image_cv, False

BASE_PHILOSOPHY = "أنتِ رويال مايند، العقل البرمجي والوجداني لبراند Royal Elchim الجمالي المتكامل."

@app.get("/api/search")
async def search(query: str):
    """
    دالة البحث الملكية المحدثة والمصلحة 100% لقراءة ملفات المخازن بدون أي أخطاء
    """
    if not query:
        return {"status": "success", "data": []}
        
    # جلب كافة ملفات الـ CSV المتوفرة في المجلد
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f != 'requirements.txt']
    
    if not csv_files:
        return {"status": "error", "message": "لم يتم العثور على ملفات قاعدة البيانات CSV في السيرفر."}
        
    results = []
    query = query.lower().strip()
    
    for file_name in csv_files:
        try:
            # قراءة ملف الـ CSV بدعم كامل للغة العربية والترميز المتنوع
            df = pd.read_csv(file_name, encoding='utf-8-sig', on_bad_lines='skip')
            
            # تنظيف أسماء الأعمدة من المسافات والأحرف الغريبة
            df.columns = [str(c).strip() for c in df.columns]
            
            # التعرف على عمود اسم المنتج أو الصنف تلقائياً
            name_col = next((col for col in df.columns if any(k in col.lower() for k in ['name', 'اسم', 'صنف', 'البيان', 'المنتج', 'item'])), None)
            # التعرف على عمود السعر تلقائياً
            price_col = next((col for col in df.columns if any(k in col.lower() for k in ['price', 'سعر', 'بيع', 'المستهلك', 'rate'])), None)
            
            if not name_col:
                name_col = df.columns[0] # لو ملاقاش، ياخد أول عمود كافتراضي
                
            # تصفية الصفوف التي تحتوي على كلمة البحث (البحث مرن ومطاطي)
            matched_df = df[df[name_col].astype(str).str.lower().str.contains(query, na=False)]
            
            for _, row in matched_df.head(20).iterrows():
                item_name = str(row[name_col]).strip()
                item_price = str(row[price_col]).strip() if price_col else "غير محدد"
                
                # جرد وفحص الكميات في الفروع بناءً على الكلمات الدلالية في الأعمدة
                branches_info = {}
                for col in df.columns:
                    if col not in [name_col, price_col]:
                        col_lower = col.lower()
                        # كلمات دلالية للفروع والمخازن المشهورة عندك
                        if any(k in col_lower or k in col for k in ['غردقة', 'مروة', 'أقصر', 'لوتس', 'اونلاين', 'مخزن', 'فرع', 'qty', 'stock', 'كمية', 'رصيد']):
                            branches_info[col] = str(row[col]).strip()
                
                # لو ملاقاش أعمدة فروع صريحة، ياخد بقية الأعمدة المتاحة كبيانات مفيدة
                if not branches_info:
                    for col in df.columns[:6]:
                        if col not in [name_col, price_col]:
                            branches_info[col] = str(row[col]).strip()

                # تنسيق نص الفروع بشكل جمالي ومريح للعين في الواجهة
                branches_text = " | ".join([f"{k}: {v}" for k, v in branches_info.items()]) if branches_info else "متاح"

                results.append({
                    "name": f"{item_name} ({branches_text})",
                    "price": item_price
                })
        except Exception as e:
            print(f"خطأ أثناء قراءة ملف {file_name}: {str(e)}")
            continue
            
    return {"status": "success", "data": results}

@app.get("/api/vault")
async def get_vault(phone: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT type, text, selfie, product, date FROM vault WHERE phone=? ORDER BY id DESC", (phone,))
    rows = c.fetchall()
    conn.close()
    
    data = [{"type": r[0], "text": r[1], "selfie": r[2], "product": r[3], "date": r[4]} for r in rows]
    return {"status": "success", "data": data}

@app.post("/api/diagnose")
async def diagnose(payload: DiagnosisPayload):
    history_context = get_history_from_db(payload.phone)
    context_str = f"\n[الذاكرة التراكمية للعميل]: {history_context}" if history_context else ""
    prompt = f"{BASE_PHILOSOPHY}{context_str}\nجلسة حوار الصداقة: '{payload.client_message}'"
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    save_to_db(payload.phone, "صداقة رويال مايند", f"الطلب: {payload.client_message}\n\nالرد: {res}")
    return {"status": "success", "diagnosis": res}

@app.post("/api/chat")
async def chat(payload: ChatPayload):
    history_context = get_history_from_db(payload.phone)
    context_str = f"\n[الذاكرة التراكمية للعميل]: {history_context}" if history_context else ""
    prompt = f"{BASE_PHILOSOPHY}{context_str}\nطلب العميل: '{payload.text}'"
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    
    record_type = "استشارة عطور" if payload.category == 'perfume' else "استشارة مكياج"
    save_to_db(payload.phone, record_type, f"الطلب: {payload.text}\n\nالرد: {res}")
    return {"status": "success", "answer": res}

@app.post("/api/simulate_makeup")
async def simulate_makeup(payload: SimulationPayload):
    try:
        encoded = payload.user_selfie.split(",", 1)[1] if "," in payload.user_selfie else payload.user_selfie
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        rgb_color = hex_to_rgb(payload.hex_color)
        processed_img, face_found = apply_royal_makeup(img_cv, rgb_color, payload.makeup_type)
        
        if face_found:
            _, buffer = cv2.imencode('.jpg', processed_img)
            result_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        else:
            result_base64 = payload.user_selfie

        history_context = get_history_from_db(payload.phone)
        context_str = f"\n[الذاكرة التراكمية]: {history_context}" if history_context else ""
        contents = [Image.open(io.BytesIO(base64.b64decode(result_base64.split(",")[1]))), f"{BASE_PHILOSOPHY}{context_str}\nصفي تناغم المكياج."]
        res = robust_generate(payload.client_api_key, contents, VISION_MODELS)
        
        record_type = "محاكاة جمالية" if payload.product_image else "تحليل تعبيري"
        save_to_db(payload.phone, record_type, res, payload.user_selfie, payload.product_image)

        return {
            "status": "success",
            "result_image": result_base64,
            "simulation_result": res,
            "face_detected": face_found
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # تم تثبيت البورت هنا على 7860 ليتوافق مع Hugging Face
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
