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
from typing import Optional, List, Dict
import time
import numpy as np
import cv2
import urllib.request
import sqlite3
from datetime import datetime

# ---------------------------------------------------------
# [استدعاء حواس المستقبل - MediaPipe Tasks API]
# ---------------------------------------------------------
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

# =========================================================
# [السجلات الملكية - قاعدة بيانات الذاكرة برقم الهاتف]
# =========================================================
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
    print("رويال مايند: جاري تحميل خريطة الوعي البصري (Face Landmarker)...")
    urllib.request.urlretrieve(TASK_URL, TASK_FILE)

base_options = python.BaseOptions(model_asset_path=TASK_FILE)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

keys_string = os.environ.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEYS", ""))
SYSTEM_API_KEYS = [key.strip() for key in keys_string.split(",") if key.strip()]

VISION_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]
TEXT_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

def get_inventory():
    try:
        file_path = "last.xls - Sheet1 (4).csv"
        if os.path.exists(file_path):
            return pd.read_csv(file_path).fillna("")
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def get_links_db():
    try:
        file_path = "Royal_Elchim_Final_Database.csv"
        if os.path.exists(file_path):
            return pd.read_csv(file_path).fillna("")
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def robust_generate(client_api_key, contents, models_list):
    if client_api_key and client_api_key.strip():
        keys_to_use = [client_api_key.strip()]
    else:
        if not SYSTEM_API_KEYS:
            raise HTTPException(status_code=500, detail="مفاتيح الخادم السحابي غير مهيأة بعد.")
        keys_to_use = SYSTEM_API_KEYS.copy()
        random.shuffle(keys_to_use)

    last_error = ""
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
                    last_error = error_str
                    print(f"👑 Royal Mind Gemini API Error ({model_name}): {error_str}") # السطر ده هيكشف المستور في الـ Logs
                    if "503" in error_str or "ResourceExhausted" in error_str or "429" in error_str:
                        time.sleep(1.5)
                        continue
                    else:
                        break
                        
    # هنعرض الخطأ التقني الحقيقي هنا عشان نعرف نعالجه فوراً
    raise HTTPException(status_code=503, detail=f"تعذر الاتصال بالذكاء الاصطناعي. الخطأ التقني: {last_error}")
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

class InvoiceItem(BaseModel):
    barcode: str
    name: str
    qty: int
    price_card_1: float
    price_card_2: float
    price_card_3: float
    price_card_4: float
    is_fixed_price: bool

class InvoicePayload(BaseModel):
    items: List[InvoiceItem]

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
            ],
            "blush": [
                [116, 117, 118, 119, 100, 120, 121, 147, 213, 192, 214, 210, 211, 32, 208, 199], 
                [345, 346, 347, 348, 329, 350, 351, 376, 433, 416, 434, 430, 431, 262, 428, 420] 
            ],
            "concealer": [
                [227, 137, 177, 215, 138, 135, 169, 170, 140, 171, 175, 199], 
                [447, 366, 401, 435, 367, 364, 394, 395, 369, 396, 400, 420]  
            ],
            "foundation": [[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]]
        }

        if makeup_type == "powder": makeup_type = "foundation"
        target_zones = ZONES.get(makeup_type, ZONES["lips"])
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for zone in target_zones:
            points = np.array([ [int(face_landmarks[idx].x * width), int(face_landmarks[idx].y * height)] for idx in zone ], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

        blur_radius = (15, 15)
        opacity = 0.6
        if makeup_type == "blush": blur_radius = (45, 45); opacity = 0.4
        elif makeup_type == "eyeshadow": blur_radius = (21, 21); opacity = 0.5
        elif makeup_type == "foundation": blur_radius = (55, 55); opacity = 0.15 
        elif makeup_type == "concealer": blur_radius = (25, 25); opacity = 0.7 

        mask = cv2.GaussianBlur(mask, blur_radius, 0)
        color_layer = np.zeros_like(image_cv)
        color_layer[:] = color_rgb[::-1]
        alpha = mask / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        blended_layer = cv2.addWeighted(image_cv, 1.0 - opacity, color_layer, opacity, 0)
        final_image = (1.0 - alpha) * image_cv + alpha * blended_layer

        return final_image.astype(np.uint8), True
    except Exception as e:
        return image_cv, False

BASE_PHILOSOPHY = "أنتِ رويال مايند، العقل البرمجي والوجداني لبراند Royal Elchim الجمالي المتكامل."

def sanitize_value(val, default_text="---"):
    if val is None: return default_text
    s = str(val).strip()
    if s.lower() == 'nan' or s == '': return default_text
    return s

def clean_qty_value(val):
    if val is None: return 0.0
    s = str(val).strip().replace(',', '.')
    if s == '' or s.lower() == 'nan': return 0.0
    try: return float(s)
    except: return 0.0

def get_qty_by_keyword(row, keywords):
    for col in row.keys():
        for kw in keywords:
            if kw in str(col):
                return clean_qty_value(row[col])
    return 0.0

@app.get("/api/search")
async def search(query: str):
    try:
        inv = get_inventory()
        db = get_links_db()
        if inv.empty: return {"status": "error", "message": "قاعدة بيانات المعرض غير متوفرة."}

        results = inv[
            inv['الصنف'].astype(str).str.contains(query, na=False, case=False, regex=False) | 
            inv['الباركود'].astype(str).str.contains(query, na=False, regex=False)
        ].head(15)

        data = []
        for _, row in results.iterrows():
            item_name = str(row.get('الصنف', '')).strip()
            is_oil = any(kw in item_name.lower() for kw in ["زيت", "جرام", "تركيب", "كحول", "مثبت"])
            qty_luxor_lotus = get_qty_by_keyword(row, ['اللوتس'])
            qty_marrowa = get_qty_by_keyword(row, ['المروة'])
            qty_hurgada = get_qty_by_keyword(row, ['HURGADA', 'الغردقة'])
            qty_online = get_qty_by_keyword(row, ['اونلاين', 'online'])

            price_1 = clean_qty_value(row.get('سعر1 كارت', 0))
            price_2 = clean_qty_value(row.get('سعر2 كارت', price_1 * 0.9))
            price_3 = clean_qty_value(row.get('سعر3 كارت', price_1 * 0.85))
            price_4 = clean_qty_value(row.get('سعر4 كارت', price_1 * 0.8))

            is_fixed = any(kw in item_name for kw in ["ثابت", "محمي", "صافي"])

            link = "https://www.royalelchim.app"
            show_link_trigger = False

            if is_oil:
                luxor_lotus_final = 0
                marrowa_final = int(qty_marrowa)
                hurgada_final = int(qty_hurgada)
            else:
                luxor_lotus_final = int(qty_luxor_lotus)
                marrowa_final = int(qty_marrowa)
                hurgada_final = int(qty_hurgada)
                link_match = db[db['Product_Name'].astype(str).str.contains(item_name, na=False, case=False, regex=False)] if not db.empty else pd.DataFrame()
                link = link_match['Product_Link'].values[0] if not link_match.empty else "https://www.royalelchim.app"
                show_link_trigger = True if qty_online > 0 else False

            data.append({
                "name": item_name,
                "price": price_1,
                "price_card_1": price_1,
                "price_card_2": price_2,
                "price_card_3": price_3,
                "price_card_4": price_4,
                "is_fixed_price": is_fixed,
                "barcode": sanitize_value(row.get('الباركود'), "---"),
                "link": sanitize_value(link, "https://www.royalelchim.app"),
                "is_oil": is_oil,
                "show_link": show_link_trigger,
                "luxor_lotus_qty": luxor_lotus_final,
                "marrowa_qty": marrowa_final,
                "hurgada_qty": hurgada_final,
                "online_qty": int(qty_online)
            })
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": f"خطأ داخلي: {str(e)}"}

@app.post("/api/invoice/calculate")
async def calculate_invoice(payload: InvoicePayload):
    try:
        initial_total = 0
        for item in payload.items:
            initial_total += item.price_card_1 * item.qty
            
        target_tier = 1
        tier_name = "قطاعي"
        if initial_total >= 30000:
            target_tier = 4
            tier_name = "جملة كبار العملاء الملكي (السعر الرابع)"
        elif initial_total >= 15000:
            target_tier = 3
            tier_name = "جملة خاصة"
        elif initial_total >= 5000:
            target_tier = 2
            tier_name = "جملة عادية"

        final_items = []
        final_invoice_total = 0

        for item in payload.items:
            if item.is_fixed_price:
                actual_price = item.price_card_1
                is_protected = True
            else:
                if target_tier == 1: actual_price = item.price_card_1
                elif target_tier == 2: actual_price = item.price_card_2
                elif target_tier == 3: actual_price = item.price_card_3
                elif target_tier == 4: actual_price = item.price_card_4
                is_protected = False

            item_total = actual_price * item.qty
            final_invoice_total += item_total

            final_items.append({
                "barcode": item.barcode,
                "name": item.name,
                "qty": item.qty,
                "applied_price": actual_price,
                "is_protected": is_protected,
                "total": item_total
            })

        return {
            "status": "success",
            "initial_total": initial_total,
            "final_total": final_invoice_total,
            "applied_tier": target_tier,
            "tier_name": tier_name,
            "items": final_items
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/vault")
async def get_vault(phone: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT type, text, selfie, product, date FROM vault WHERE phone=? ORDER BY id DESC", (phone,))
    rows = c.fetchall()
    conn.close()
    
    data = [{"type": r[0], "text": r[1], "selfie": r[2], "product": r[3], "date": r[4]} for r in rows]
    return {"status": "success", "data": data}

# --- استدعاء الذاكرة برقم الهاتف في كل المحادثات والتشخيص ---
@app.post("/api/diagnose")
async def diagnose(payload: DiagnosisPayload):
    history_context = get_history_from_db(payload.phone)
    context_str = f"\n[الذاكرة التراكمية للعميل - رقم {payload.phone}]: {history_context}" if history_context else ""
    prompt = f"{BASE_PHILOSOPHY}{context_str}\nجلسة حوار الصداقة والتحليل النفسي: '{payload.client_message}'"
    
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    save_to_db(payload.phone, "صداقة رويال مايند", f"الطلب: {payload.client_message}\n\nالرد: {res}")
    
    return {"status": "success", "diagnosis": res}

@app.post("/api/chat")
async def chat(payload: ChatPayload):
    history_context = get_history_from_db(payload.phone)
    context_str = f"\n[الذاكرة التراكمية للعميل - رقم {payload.phone}]: {history_context}" if history_context else ""
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

        # استدعاء الذاكرة لربط التخيل البصري برقم الهاتف
        history_context = get_history_from_db(payload.phone)
        context_str = f"\n[الذاكرة التراكمية للعميل - رقم {payload.phone}]: {history_context}" if history_context else ""
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
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
