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
from datetime import datetime

import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 📚 مكتبة قراءة الـ PDF السحرية
try:
    import PyPDF2
except ImportError:
    pass

app = FastAPI(title="Royal Elchim - Omni-Conscious Enterprise")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "👑 Royal Mind Engine is running successfully on Cloud Vault!"}

# =========================================================
# [السجلات الملكية - MongoDB Cloud Vault]
# =========================================================
MONGO_URI = os.environ.get("MONGO_URI", "")
mongo_client = None
vault_collection = None

if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client["royal_engine"]
        vault_collection = db["vault"]
    except Exception as e:
        print(f"خطأ في الاتصال بـ MongoDB: {e}")

def save_to_db(phone, record_type, text, selfie=None, product=None):
    if not phone or vault_collection is None: return
    clean_phone = str(phone).strip()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if selfie:
        existing_selfies = list(vault_collection.find(
            {"phone": {"$eq": clean_phone}, "selfie": {"$ne": None}}
        ).sort("date", 1))
        
        if len(existing_selfies) >= 5:
            for i in range(len(existing_selfies) - 4):
                vault_collection.delete_one({"_id": existing_selfies[i]["_id"]})

    document = {
        "phone": clean_phone,
        "type": record_type,
        "text": text,
        "selfie": selfie,
        "product": product,
        "date": date_str
    }
    vault_collection.insert_one(document)

def get_history_from_db(phone, limit=10):
    if not phone or vault_collection is None: return ""
    clean_phone = str(phone).strip()
    records = list(vault_collection.find({"phone": {"$eq": clean_phone}}).sort("date", -1).limit(limit))
    if not records: return ""
    history = " | ".join([f"[تاريخ {r['date']} - {r['type']}]: {r['text']}" for r in reversed(records)])
    return history

TASK_FILE = 'face_landmarker.task'
TASK_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

if not os.path.exists(TASK_FILE):
    urllib.request.urlretrieve(TASK_URL, TASK_FILE)

base_options = python.BaseOptions(model_asset_path=TASK_FILE)
options = vision.FaceLandmarkerOptions(
    base_options=base_options, num_faces=1, min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
    output_face_blendshapes=False, output_facial_transformation_matrixes=False
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

keys_string = os.environ.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEYS", ""))
SYSTEM_API_KEYS = [key.strip() for key in keys_string.split(",") if key.strip()]
TEXT_MODELS = ["gemini-3.5-pro", "gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
VISION_MODELS = TEXT_MODELS

# =========================================================
# 🛠️ [NATIVE FUNCTION CALLING TOOLS] الأسلحة الذكية
# =========================================================
def read_local_file(filename: str) -> str:
    """
    أداة لقراءة الملفات (سواء PDF أو TXT أو CSV) من خوادم Royal Elchim.
    قم باستدعاء هذه الأداة حصرياً إذا طلب منك العميل قراءة ملف معين أو البحث عن تركيبة كيميائية معقدة داخل ملف.
    """
    try:
        target_file = None
        for f in os.listdir('.'):
            if filename.lower() in f.lower():
                target_file = f
                break
                
        if not target_file:
            return f"عذراً، لم أتمكن من العثور على ملف باسم '{filename}' في الخزنة السرية."

        # إذا كان الملف PDF
        if target_file.lower().endswith('.pdf'):
            text = f"--- محتوى ملف {target_file} ---\n"
            with open(target_file, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text[:20000] # حماية من الحجم الزائد

        # إذا كان ملف نصي عادي
        else:
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                return f"--- محتوى ملف {target_file} ---\n" + f.read()[:20000]
                
    except Exception as e:
        return f"حدث خطأ أثناء محاولة قراءة الملف: {str(e)}"

# =========================================================
# [محرك التوليد المدعوم بـ Tool Calling]
# =========================================================
def robust_generate(client_api_key, contents, models_list):
    keys_to_use = [client_api_key.strip()] if client_api_key and client_api_key.strip() else SYSTEM_API_KEYS.copy()
    if not keys_to_use: raise HTTPException(status_code=500, detail="مفاتيح الخادم السحابي غير مهيأة بعد.")
    random.shuffle(keys_to_use)
    last_error = ""
    for model_name in models_list:
        for key in keys_to_use:
            for _ in range(2):
                try:
                    client = genai.Client(api_key=key)
                    # 🎯 زراعة الأداة بداخل الـ Config ليستخدمها الذكاء تلقائياً
                    config = types.GenerateContentConfig(
                        temperature=0.8, 
                        top_p=0.95,
                        tools=[read_local_file] 
                    )
                    
                    # نستخدم واجهة الـ Chat لأنها تنفذ الـ Function Calling تلقائياً 
                    # (الذكاء الاصطناعي يطلب الأداة -> السيرفر ينفذها -> يعيد الناتج للذكاء -> الذكاء يرد عليك)
                    chat = client.chats.create(model=model_name, config=config)
                    response = chat.send_message(contents)
                    
                    if response and response.text: 
                        return response.text
                except Exception as e:
                    last_error = str(e)
                    if any(x in last_error for x in ["503", "ResourceExhausted", "429"]): time.sleep(1.5); continue
                    break
    raise HTTPException(status_code=503, detail=f"تعذر الاتصال بالذكاء الاصطناعي. الخطأ: {last_error}")

def get_inventory():
    try:
        files = [f for f in os.listdir('.') if f.endswith('.csv')]
        target_file = None
        for f in files:
            if 'last' in f.lower() or 'sheet' in f.lower():
                target_file = f
                break
        if not target_file and files: target_file = files[0]
        if target_file:
            return pd.read_csv(target_file, encoding='utf-8-sig', on_bad_lines='skip').fillna("")
        return pd.DataFrame()
    except: return pd.DataFrame()

def get_brand_catalog():
    catalog = ""
    try:
        for f in os.listdir('.'):
            if f.endswith('.txt') and 'requirements' not in f.lower():
                with open(f, 'r', encoding='utf-8') as file:
                    catalog += file.read() + "\n"
        brand_file = "ROYALELCHIMBRAND.xls.xlsx"
        if os.path.exists(brand_file):
            df = pd.read_excel(brand_file)
            if 'الصنف' in df.columns:
                ready_items = [str(row.get('الصنف', '')).strip() for _, row in df.head(135).iterrows() if str(row.get('الصنف', '')).strip() and str(row.get('الصنف', '')).lower() != 'nan']
                if ready_items:
                    catalog += "\n[قائمة عطور براند Royal Elchim الجاهزة في المعرض - للترشيح المباشر]:\n" + "، ".join(ready_items) + "\n"
        return catalog
    except: return catalog

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

class PerfumeCraftPayload(BaseModel):
    client_message: str
    phone: str
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
        if not detection_result.face_landmarks: return image_cv, False

        height, width, _ = image_cv.shape
        face_landmarks = detection_result.face_landmarks[0]
        ZONES = {
            "lips": [[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185]],
            "eyeshadow": [[33, 246, 161, 160, 159, 158, 157, 173, 133], [362, 398, 384, 385, 386, 387, 388, 466, 263]],
            "blush": [[116, 117, 118, 119, 100, 120, 121, 147, 213, 192, 214, 210, 211, 32, 208, 199], [345, 346, 347, 348, 329, 350, 351, 376, 433, 416, 434, 430, 431, 262, 428, 420]],
            "highlighter": [[116, 117, 118, 119, 147, 213], [345, 346, 347, 348, 376, 433], [197, 195, 5, 4]],
            "foundation": [[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]]
        }
        target_zones = ZONES.get(makeup_type, ZONES["lips"])
        mask = np.zeros((height, width), dtype=np.uint8)
        for zone in target_zones:
            points = np.array([ [int(face_landmarks[idx].x * width), int(face_landmarks[idx].y * height)] for idx in zone ], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        blur_radius = (45, 45) if makeup_type in ["blush", "foundation"] else (21, 21)
        mask = cv2.GaussianBlur(mask, blur_radius, 0)
        alpha = np.expand_dims(mask / 255.0, axis=-1)
        color_layer = np.zeros_like(image_cv)
        color_layer[:] = color_rgb[::-1]

        if makeup_type == "foundation":
            smooth_skin = cv2.bilateralFilter(image_cv, 15, 75, 75)
            blended_layer = cv2.addWeighted(smooth_skin, 0.7, color_layer, 0.3, 0)
        elif makeup_type == "highlighter":
            blended_layer = cv2.addWeighted(image_cv, 0.5, cv2.add(image_cv, color_layer), 0.5, 0)
        else:
            multiply_blend = (image_cv.astype(np.float32) * color_layer.astype(np.float32)) / 255.0
            opacity = 0.55 if makeup_type == "lips" else 0.4
            blended_layer = ((1.0 - opacity) * image_cv.astype(np.float32) + opacity * multiply_blend).astype(np.uint8)
        return ((1.0 - alpha) * image_cv + alpha * blended_layer).astype(np.uint8), True
    except: return image_cv, False

BASE_PHILOSOPHY = "أنتِ رويال مايند، العقل البرمجي والوجداني لبراند Royal Elchim الجمالي المتكامل."

def clean_qty_value(val):
    if val is None: return 0.0
    s = str(val).strip().replace(',', '.')
    if s == '' or s.lower() == 'nan': return 0.0
    try: return float(s)
    except: return 0.0

def get_qty_by_keyword(row, keywords):
    for col in row.keys():
        for kw in keywords:
            if kw in str(col).lower():
                return clean_qty_value(row[col])
    return 0.0

@app.get("/api/search")
async def search(query: str):
    try:
        inv = get_inventory()
        if inv.empty or 'الصنف' not in inv.columns: 
            return {"status": "error", "message": "ملف قاعدة بيانات الجرد الحقيقي غير متوفر."}

        query_words = query.strip().split()
        mask = pd.Series([True] * len(inv))
        for word in query_words:
            mask = mask & (inv['الصنف'].astype(str).str.contains(word, case=False, na=False) | inv['الباركود'].astype(str).str.contains(word, case=False, na=False))
        
        results = inv[mask].head(20)
        data = []
        for _, row in results.iterrows():
            item_name = str(row.get('الصنف', '')).strip()
            qty_luxor_lotus = get_qty_by_keyword(row, ['اللوتس', 'لوتس', 'اقصر', 'أقصر', 'luxor'])
            qty_marrowa = get_qty_by_keyword(row, ['المروة', 'المروه', 'مروة', 'مروه', 'marwa'])
            qty_hurgada = get_qty_by_keyword(row, ['الغردقة', 'الغردقه', 'غردقة', 'غردقه', 'hurgada', 'hurghada'])
            qty_online = get_qty_by_keyword(row, ['اونلاين', 'online', 'أونلاين', 'مخزن'])

            is_oil = any(kw in item_name.lower() for kw in ["زيت", "جرام", "تركيب", "كحول", "مثبت"])
            if is_oil:
                luxor_lotus_final = 0
                marrowa_final = int(qty_marrowa)
                hurgada_final = int(qty_hurgada)
            else:
                luxor_lotus_final = int(qty_luxor_lotus)
                marrowa_final = int(qty_marrowa)
                hurgada_final = int(qty_hurgada)

            price_1 = clean_qty_value(row.get('سعر1 كارت', 0))
            branches_text = f"🟢 المروة: {marrowa_final} | 🔵 الغردقة: {hurgada_final} | 🟡 الأقصر: {luxor_lotus_final} | 🌐 أونلاين: {int(qty_online)}"
            full_name_with_branches = f"{item_name} <br> <span class='text-muted small'>📦 ( {branches_text} )</span>"
            data.append({"name": full_name_with_branches, "price": price_1})
            
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": f"خطأ داخلي: {str(e)}"}

@app.get("/api/vault")
async def get_vault(phone: str):
    if vault_collection is None: return {"status": "error", "message": "قاعدة البيانات غير متصلة"}
    clean_phone = str(phone).strip()
    records = list(vault_collection.find({"phone": {"$eq": clean_phone}}).sort("date", -1))
    
    data = []
    for r in records:
        if str(r.get('phone')).strip() == clean_phone:
            data.append({
                "id": str(r.get('_id')), 
                "type": r.get('type'), 
                "text": r.get('text'), 
                "selfie": r.get('selfie'), 
                "product": r.get('product'), 
                "date": r.get('date')
            })
    return {"status": "success", "data": data}

@app.delete("/api/vault/{record_id}")
async def delete_vault_record(record_id: str, phone: str):
    if vault_collection is None: return {"status": "error"}
    try:
        result = vault_collection.delete_one({"_id": ObjectId(record_id), "phone": str(phone).strip()})
        if result.deleted_count > 0:
            return {"status": "success"}
        return {"status": "error", "message": "Record not found"}
    except:
        return {"status": "error"}

@app.post("/api/diagnose")
async def diagnose(payload: DiagnosisPayload):
    history = get_history_from_db(payload.phone)
    profiler = f"تاريخ العميل: [{history}]، استنتجي بصمته الوجدانية." if history else "عميل جديد."
    # 🎯 تعليمات واضحة للذكاء ليعرف أنه يملك الأداة
    prompt = f"{BASE_PHILOSOPHY}\n\n{profiler}\nملاحظة لك كذكاء اصطناعي: أنت تملك أداة اسمها read_local_file، إذا طلب العميل قراءة ملف PDF أو نصي قم باستخدامها فوراً.\n\nطلب العميل: '{payload.client_message}'"
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    save_to_db(payload.phone, "صداقة رويال مايند", f"الطلب: {payload.client_message}\n\nالرد: {res}")
    return {"status": "success", "diagnosis": res, "answer": res}

@app.post("/api/chat")
async def chat(payload: ChatPayload):
    history = get_history_from_db(payload.phone)
    profiler = f"تاريخ العميل: [{history}]، استنتجي بصمته الوجدانية." if history else "عميل جديد."
    prompt = f"{BASE_PHILOSOPHY}\n\n{profiler}\nملاحظة لك كذكاء اصطناعي: أنت تملك أداة اسمها read_local_file، إذا طلب العميل قراءة ملف PDF أو نصي قم باستخدامها فوراً.\n\nطلب العميل: '{payload.text}'"
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    save_to_db(payload.phone, "استشارة مكياج", f"الطلب: {payload.text}\n\nالرد: {res}")
    return {"status": "success", "diagnosis": res, "answer": res}

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
        makeup_names_ar = {"lips": "أحمر الشفاه", "eyeshadow": "ظلال العيون", "blush": "أحمر الخدود", "concealer": "الكونسيلر", "foundation": "كريم الأساس", "highlighter": "الهايلايتر"}
        makeup_name = makeup_names_ar.get(payload.makeup_type, "المكياج")
        
        expert_prompt = f"{BASE_PHILOSOPHY}\nالتاريخ: [{history_context}]\nالعميلة جربت '{makeup_name}'. حللي كيف اندمج اللون وقدمي نصيحة."
        contents = [Image.open(io.BytesIO(base64.b64decode(result_base64.split(",")[1]))), expert_prompt]
                
        res = robust_generate(payload.client_api_key, contents, VISION_MODELS)
        save_to_db(payload.phone, f"محاكاة {makeup_name}", res, result_base64, payload.product_image)

        return {"status": "success", "result_image": result_base64, "simulation_result": res, "face_detected": face_found}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/craft_perfume")
async def craft_perfume(payload: PerfumeCraftPayload):
    try:
        inv = get_inventory()
        brand_catalog = get_brand_catalog()
        
        available_oils = []
        if not inv.empty and 'الصنف' in inv.columns:
            oils_df = inv[inv['الصنف'].astype(str).str.contains("SAVVY|PARFUME OIL", case=False, na=False)]
            for _, row in oils_df.head(60).iterrows():
                name = str(row.get('الصنف', '')).strip()
                price = float(str(row.get('سعر1 كارت', 0)).replace(',','.')) if str(row.get('سعر1 كارت', 0)).strip() != '' else 0
                if price > 0: available_oils.append(f"{name} (السعر: {price} ج.م/للجرام)")
        
        oils_list_text = " \n ".join(available_oils) if available_oils else "قاعدة الأسعار قيد التحديث."
        history_context = get_history_from_db(payload.phone)
        
        perfume_prompt = (
            f"{BASE_PHILOSOPHY}\nالتاريخ التراكمي: [{history_context}]\n"
            f"ملاحظة لك كذكاء اصطناعي: أنت تملك أداة اسمها read_local_file، إذا طلب العميل قراءة تركيبة من ملف PDF أو نصي موجود على السيرفر قم باستدعائها.\n\n"
            f"رسالة العميل: '{payload.client_message}'\n\n"
            f"--- القوائم ---\n1. [زيوت التركيب الخام بأسعارها للتركيب فقط]:\n{oils_list_text}\n\n"
            f"2. [أرشيف عطور Royal Elchim الجاهزة للترشيح الجاهز فقط]:\n{brand_catalog}\n-----------------\n\n"
            f"المهام بصرامة:\n"
            f"1. استنتجي البصمة الوجدانية للعميل واربطيها بقاعدة العطر الأساسي ROYAL E.K.A.\n"
            f"2. صممي تركيبة لزجاجة 50 مل باستخدام [زيوت التركيب الخام] فقط. واربطيها ببرج فلكي وحيوان روحي.\n"
            f"3. احسبي تكلفة الزيوت، والمثبت (1 جم لكل 5 جم زيت بسعر 3 ج.م)، والكحول (0.5 ج.م للجرام).\n"
            f"4. يُمنع منعاً باتاً ترشيح أي زيت خام كعطر جاهز. يجب أن ترشحي عطراً جاهزاً مأخوذاً حصرياً وبالنص من [أرشيف عطور Royal Elchim الجاهزة]. اشرحي لماذا يناسب شخصيته كبديل فوري.\n"
            f"5. كوني كائناً حياً صديقاً."
        )

        res = robust_generate(payload.client_api_key, [perfume_prompt], TEXT_MODELS)
        save_to_db(payload.phone, "تصميم عطر ملكي شامل", f"الطلب: {payload.client_message}\n\nالتركيبة: {res}")
        return {"status": "success", "answer": res, "diagnosis": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
