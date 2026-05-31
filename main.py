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

# استدعاء مكتبة MongoDB للخزنة السحابية الأبدية
import pymongo
from pymongo import MongoClient

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
# [الباب الرئيسي]
# =========================================================
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
        print("رويال مايند: تم الاتصال بالخزنة السحابية MongoDB بنجاح!")
    except Exception as e:
        print(f"خطأ في الاتصال بـ MongoDB: {e}")

def save_to_db(phone, record_type, text, selfie=None, product=None):
    if not phone or vault_collection is None: return
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # حد أقصى 5 صور للحفاظ على كفاءة الذاكرة
    if selfie:
        existing_selfies = list(vault_collection.find(
            {"phone": phone, "selfie": {"$ne": None}}
        ).sort("date", 1))
        
        if len(existing_selfies) >= 5:
            records_to_delete = len(existing_selfies) - 4
            for i in range(records_to_delete):
                vault_collection.delete_one({"_id": existing_selfies[i]["_id"]})

    document = {
        "phone": phone,
        "type": record_type,
        "text": text,
        "selfie": selfie,
        "product": product,
        "date": date_str
    }
    vault_collection.insert_one(document)

def get_history_from_db(phone, limit=5):
    if not phone or vault_collection is None: return ""
    records = list(vault_collection.find({"phone": phone}).sort("date", -1).limit(limit))
    if not records: return ""
    history = " | ".join([f"[تاريخ {r['date']} - {r['type']}]: {r['text']}" for r in reversed(records)])
    return history

# =========================================================
# [الوعي البصري والموديلات - القوة الضاربة والاحتياطي]
# =========================================================
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
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

keys_string = os.environ.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEYS", ""))
SYSTEM_API_KEYS = [key.strip() for key in keys_string.split(",") if key.strip()]

VISION_MODELS = [
    "gemini-3.5-pro", 
    "gemini-3.5-flash", 
    "gemini-2.5-pro", 
    "gemini-2.5-flash", 
    "gemini-1.5-pro", 
    "gemini-1.5-flash"
]

TEXT_MODELS = [
    "gemini-3.5-pro", 
    "gemini-3.5-flash", 
    "gemini-2.5-pro", 
    "gemini-2.5-flash", 
    "gemini-1.5-pro", 
    "gemini-1.5-flash"
]

# =========================================================
# [جلب الملفات بمرونة]
# =========================================================
def get_inventory():
    try:
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and ('last' in f.lower() or 'sheet1' in f.lower())]
        file_path = csv_files[0] if csv_files else "last.xls - Sheet1.csv"
        if os.path.exists(file_path):
            return pd.read_csv(file_path, encoding='utf-8-sig', on_bad_lines='skip').fillna("")
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def get_links_db():
    try:
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'database' in f.lower()]
        file_path = csv_files[0] if csv_files else "Royal_Elchim_Final_Database.csv"
        if os.path.exists(file_path):
            return pd.read_csv(file_path, encoding='utf-8-sig', on_bad_lines='skip').fillna("")
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
                    if "503" in error_str or "ResourceExhausted" in error_str or "429" in error_str:
                        time.sleep(1.5)
                        continue
                    else:
                        break
    raise HTTPException(status_code=503, detail=f"تعذر الاتصال بالذكاء الاصطناعي. الخطأ: {last_error}")

# =========================================================
# [هياكل البيانات]
# =========================================================
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

# =========================================================
# [معالجة الصور والمكياج بواقعية سينمائية فائقة]
# =========================================================
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
            "highlighter": [
                [116, 117, 118, 119, 147, 213], # خد أيسر
                [345, 346, 347, 348, 376, 433], # خد أيمن
                [197, 195, 5, 4] # أرنبة الأنف
            ],
            "foundation": [[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]]
        }

        if makeup_type == "powder": makeup_type = "foundation"
        target_zones = ZONES.get(makeup_type, ZONES["lips"])
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for zone in target_zones:
            points = np.array([ [int(face_landmarks[idx].x * width), int(face_landmarks[idx].y * height)] for idx in zone ], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

        # نعومة الماسك حسب نوع المكياج
        blur_radius = (45, 45) if makeup_type in ["blush", "foundation"] else (21, 21)
        mask = cv2.GaussianBlur(mask, blur_radius, 0)
        
        alpha = mask / 255.0
        alpha = np.expand_dims(alpha, axis=-1)

        color_layer = np.zeros_like(image_cv)
        color_layer[:] = color_rgb[::-1]

        # خوارزميات الدمج الواقعية
        if makeup_type == "foundation":
            # تنعيم مسام البشرة أولاً
            smooth_skin = cv2.bilateralFilter(image_cv, 15, 75, 75)
            # دمج لون الأساس مع البشرة الناعمة
            blended_layer = cv2.addWeighted(smooth_skin, 0.7, color_layer, 0.3, 0)
        elif makeup_type == "highlighter":
            # دمج مضيء للهايلايتر
            blended_layer = cv2.add(image_cv, color_layer)
            blended_layer = cv2.addWeighted(image_cv, 0.5, blended_layer, 0.5, 0)
        else:
            # Multiply Blend للشفاه والآيشادو والبلاشر لاحتفاظ واقعي بالظلال
            image_float = image_cv.astype(np.float32)
            color_float = color_layer.astype(np.float32)
            multiply_blend = (image_float * color_float) / 255.0
            opacity = 0.55 if makeup_type == "lips" else 0.4
            blended_layer = ((1.0 - opacity) * image_float + opacity * multiply_blend).astype(np.uint8)

        final_image = (1.0 - alpha) * image_cv + alpha * blended_layer
        return final_image.astype(np.uint8), True
    except Exception as e:
        return image_cv, False

BASE_PHILOSOPHY = "أنتِ رويال مايند، العقل البرمجي والوجداني لبراند Royal Elchim الجمالي المتكامل."

# =========================================================
# [البحث الذكي والجرد والتنظيف]
# =========================================================
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
            if kw in str(col).lower():
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
            
            qty_luxor_lotus = get_qty_by_keyword(row, ['اللوتس', 'لوتس', 'اقصر', 'أقصر', 'luxor'])
            qty_marrowa = get_qty_by_keyword(row, ['المروة', 'المروه', 'مروة', 'مروه', 'marwa'])
            qty_hurgada = get_qty_by_keyword(row, ['الغردقة', 'الغردقه', 'غردقة', 'غردقه', 'hurgada', 'hurghada'])
            qty_online = get_qty_by_keyword(row, ['اونلاين', 'online', 'أونلاين', 'مخزن'])

            price_1 = clean_qty_value(row.get('سعر1 كارت', 0))
            price_2 = clean_qty_value(row.get('سعر2 كارت', price_1 * 0.9))
            price_3 = clean_qty_value(row.get('سعر3 كارت', price_1 * 0.85))
            price_4 = clean_qty_value(row.get('سعر4 كارت', price_1 * 0.8))

            is_fixed = any(kw in item_name for kw in ["ثابت", "محمي", "صافي"])

            if is_oil:
                luxor_lotus_final = 0
                marrowa_final = int(qty_marrowa)
                hurgada_final = int(qty_hurgada)
            else:
                luxor_lotus_final = int(qty_luxor_lotus)
                marrowa_final = int(qty_marrowa)
                hurgada_final = int(qty_hurgada)

            link = "https://www.royalelchim.app"
            show_link_trigger = False
            
            if not db.empty:
                link_match = db[db['Product_Name'].astype(str).str.contains(item_name, na=False, case=False, regex=False)]
                if not link_match.empty:
                    extracted_link = str(link_match['Product_Link'].values[0]).strip()
                    if extracted_link and extracted_link.startswith("http"):
                        link = extracted_link
                        show_link_trigger = True

            branches_text = f"🟢 المروة: {marrowa_final} | 🔵 الغردقة: {hurgada_final} | 🟡 الأقصر: {luxor_lotus_final} | 🌐 أونلاين: {int(qty_online)}"
            full_name_with_branches = f"{item_name}  📦 ( {branches_text} )"

            data.append({
                "name": full_name_with_branches,
                "price": price_1,
                "price_card_1": price_1,
                "price_card_2": price_2,
                "price_card_3": price_3,
                "price_card_4": price_4,
                "is_fixed_price": is_fixed,
                "barcode": sanitize_value(row.get('الباركود'), "---"),
                "link": link,
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

# =========================================================
# [النقاط النهائية للذاكرة والذكاء الاصطناعي]
# =========================================================
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
    if vault_collection is None:
        return {"status": "error", "message": "قاعدة البيانات غير متصلة"}
    records = list(vault_collection.find({"phone": phone}).sort("date", -1))
    data = [{"type": r.get('type'), "text": r.get('text'), "selfie": r.get('selfie'), "product": r.get('product'), "date": r.get('date')} for r in records]
    return {"status": "success", "data": data}

@app.post("/api/diagnose")
async def diagnose(payload: DiagnosisPayload):
    history_context = get_history_from_db(payload.phone)
    context_str = f"\n[الذاكرة التراكمية للعميل - رقم {payload.phone}]: {history_context}" if history_context else ""
    prompt = f"{BASE_PHILOSOPHY}{context_str}\nجلسة حوار الصداقة والتحليل النفسي: '{payload.client_message}'"
    
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    save_to_db(payload.phone, "صداقة رويال مايند", f"الطلب: {payload.client_message}\n\nالرد: {res}")
    
    return {"status": "success", "diagnosis": res, "answer": res}

@app.post("/api/chat")
async def chat(payload: ChatPayload):
    history_context = get_history_from_db(payload.phone)
    context_str = f"\n[الذاكرة التراكمية للعميل - رقم {payload.phone}]: {history_context}" if history_context else ""
    prompt = f"{BASE_PHILOSOPHY}{context_str}\nطلب العميل: '{payload.text}'"
    
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    record_type = "استشارة عطور" if payload.category == 'perfume' else "استشارة مكياج"
    save_to_db(payload.phone, record_type, f"الطلب: {payload.text}\n\nالرد: {res}")
    
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
        context_str = f"\n[الذاكرة التراكمية للعميل - رقم {payload.phone}]: {history_context}" if history_context else ""
        
        makeup_names_ar = {
            "lips": "أحمر الشفاه",
            "eyeshadow": "ظلال العيون (الآيشادو)",
            "blush": "أحمر الخدود (البلاشر)",
            "concealer": "الكونسيلر",
            "foundation": "كريم الأساس",
            "highlighter": "الهايلايتر اللامع"
        }
        makeup_name = makeup_names_ar.get(payload.makeup_type, "المكياج")
        product_info = payload.product_name_desc if payload.product_name_desc else 'منتج جمالي'
        
        expert_prompt = (
            f"{BASE_PHILOSOPHY}{context_str}\n"
            f"العميلة قامت الآن بتجربة افتراضية لـ '{makeup_name}' "
            f"(المنتج: {product_info}).\n"
            f"بصفتكِ خبيرة تجميل عالمية، حللي كيف اندمج هذا الـ {makeup_name} "
            f"مع ملامحها في هذه الصورة. أعطِها نصيحة احترافية مخصصة بناءً على التجربة المرئية."
        )
        
        contents = [Image.open(io.BytesIO(base64.b64decode(result_base64.split(",")[1]))), expert_prompt]
                
        res = robust_generate(payload.client_api_key, contents, VISION_MODELS)
        record_type = f"محاكاة {makeup_name}"
        save_to_db(payload.phone, record_type, res, payload.user_selfie, payload.product_image)

        return {
            "status": "success",
            "result_image": result_base64,
            "simulation_result": res,
            "face_detected": face_found
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/craft_perfume")
async def craft_perfume(payload: PerfumeCraftPayload):
    try:
        inv = get_inventory()
        if not inv.empty:
            oils_df = inv[inv['الصنف'].astype(str).str.contains("SAVVY|PARFUME OIL", case=False, na=False)]
            available_oils = oils_df['الصنف'].head(40).tolist()
            oils_list_text = "، ".join([str(x) for x in available_oils])
        else:
            oils_list_text = "قاعدة الزيوت غير متاحة حالياً."

        history_context = get_history_from_db(payload.phone)
        context_str = f"\n[الذاكرة التراكمية للعميل - رقم {payload.phone}]: {history_context}" if history_context else ""
        
        perfume_prompt = (
            f"{BASE_PHILOSOPHY}{context_str}\n"
            f"أنتِ الآن 'خيميائية العطور' الخاصة ببراند Royal Elchim.\n"
            f"رسالة العميل: '{payload.client_message}'\n\n"
            f"المهام:\n"
            f"1. اختاري مزيجاً سحرياً من هذه الزيوت المتاحة في مخازننا حصراً: [{oils_list_text}].\n"
            f"2. اربطي هذه التركيبة بشكل إبداعي بأحد الأبراج (خاصة العذراء، الجوزاء، أو الحوت) وحيوان روحي (مثل الصقر أو الدولفين) ليعكس شخصية العميل.\n"
            f"3. اقترحي اسماً راقياً، وتأكدي دائماً عند الإشارة لعطرنا الأساسي أن يكتب هكذا: ROYAL E.K.A.\n"
            f"4. قدمي نسب الخلط التقريبية بأسلوب فني ساحر."
        )

        res = robust_generate(payload.client_api_key, [perfume_prompt], TEXT_MODELS)
        save_to_db(payload.phone, "تصميم عطر ملكي", f"الطلب: {payload.client_message}\n\nالتركيبة: {res}")
        
        return {"status": "success", "answer": res, "diagnosis": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
