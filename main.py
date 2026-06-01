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

# 👑 السطر الذي كان مفقوداً وعاد لعرشه:
BASE_PHILOSOPHY = "أنتِ رويال مايند، العقل البرمجي والوجداني لبراند Royal Elchim الجمالي المتكامل."

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
# 🛠️ [NATIVE FUNCTION CALLING TOOLS] الأداة السحرية
# =========================================================
def read_local_file(filename: str) -> str:
    """
    أداة لقراءة الملفات (سواء PDF أو TXT أو CSV) من خوادم Royal Elchim.
    قم باستدعاء هذه الأداة حصرياً إذا طلب منك العميل قراءة ملف معين أو البحث عن تركيبة داخل ملف.
    """
    try:
        target_file = None
        for f in os.listdir('.'):
            if filename.lower() in f.lower():
                target_file = f
                break
                
        if not target_file:
            return f"عذراً، لم أتمكن من العثور على ملف باسم '{filename}' في الخزنة السرية."

        if target_file.lower().endswith('.pdf'):
            text = f"--- محتوى ملف {target_file} ---\n"
            with open(target_file, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text[:25000] # حماية من الحجم الزائد

        else:
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                return f"--- محتوى ملف {target_file} ---\n" + f.read()[:25000]
                
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
    
    # معالجة ذكية: تحويل المصفوفة لنص إذا كانت تحتوي على رسالة واحدة فقط (لتشغيل الأدوات)
    safe_contents = contents[0] if isinstance(contents, list) and len(contents) == 1 and isinstance(contents[0], str) else contents

    for model_name in models_list:
        for key in keys_to_use:
            for _ in range(2):
                try:
                    client = genai.Client(api_key=key)
                    config_with_tools = types.GenerateContentConfig(temperature=0.8, top_p=0.95, tools=[read_local_file])
                    config_without_tools = types.GenerateContentConfig(temperature=0.8, top_p=0.95)
                    
                    if isinstance(safe_contents, list):
                        # للصور (الواقع المعزز): لا نستخدم أدوات لمنع التعارض
                        response = client.models.generate_content(model=model_name, contents=safe_contents, config=config_without_tools)
                    else:
                        # للنصوص العادية: نستخدم وضع الدردشة ليستطيع تشغيل الأداة تلقائياً
                        chat = client.chats.create(model=model_name, config=config_with_tools)
                        response = chat.send_message(safe_contents)
                    
                    if response and response.text: 
                        return response.text
                except Exception as e:
                    last_error = str(e)
                    if any(x in last_error for x in ["503", "ResourceExhausted", "429"]): time.sleep(1.5); continue
                    break
    raise HTTPException(status_code=503, detail=f"تعذر الاتصال بالذكاء الاصطناعي. الخطأ: {last_error}")

# =========================================================
# [دوال التنظيف والقراءة]
# =========================================================
def clean_qty_value(val):
    if val is None: return 0.0
    s = str(val).strip().replace(',', '.')
    if s == '' or s.lower() == 'nan': return 0.0
    try: return float(s)
    except: return 0.0

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
    prompt = f"{BASE_PHILOSOPHY}\n\n{profiler}\nملاحظة هامة لك: أنت تملك أداة قوية اسمها read_local_file، إذا طلب العميل قراءة ملف PDF أو نصي موجود على السيرفر قم باستدعائها تلقائياً.\n\nطلب العميل: '{payload.client_message}'"
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    save_to_db(payload.phone, "صداقة رويال مايند", f"الطلب: {payload.client_message}\n\nالرد: {res}")
    return {"status": "success", "diagnosis": res, "answer": res}

@app.post("/api/chat")
async def chat(payload: ChatPayload):
    history = get_history_from_db(payload.phone)
    profiler = f"تاريخ العميل: [{history}]، استنتجي بصمته الوجدانية." if history else "عميل جديد."
    prompt = f"{BASE_PHILOSOPHY}\n\n{profiler}\nملاحظة هامة لك: أنت تملك أداة قوية اسمها read_local_file، إذا طلب العميل قراءة ملف PDF أو نصي موجود على السيرفر قم باستدعائها تلقائياً.\n\nطلب العميل: '{payload.text}'"
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    save_to_db(payload.phone, "استشارة مكياج", f"الطلب: {payload.text}\n\nالرد: {res}")
    return {"status": "success", "diagnosis": res, "answer": res}

@app.post("/api/simulate_makeup")
async def simulate_makeup(payload
