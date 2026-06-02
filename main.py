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
import urllib.parse

import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

# 👑 النبض الفلسفي والوجداني لرويال مايند
BASE_PHILOSOPHY = (
    "أنتِ رويال مايند، العقل البرمجي والوجداني لبراند Royal Elchim الجمالي المتكامل. "
    "تتحدثين كخبير تجميل وعطور وعناية بالشعر عالمي، بأسلوب فاخر، راقٍ، وصديق مقرب للعميلة."
)

def get_brand_catalog() -> str:
    return """
    كتالوج براند Royal Elchim الفاخر:
    1. ROYAL E.K.A: العطر الأساسي الإمبراطوري، يتميز بنوتات ملكية غامضة وفخمة (يجب كتابته بالنقط بين الحروف).
    2. Royal Ignite: عطر التوهج والشغف، نوتات دافئة من العنبر والتوابل الفاخرة.
    3. Royal Midnight Rose: عطر الغموض الساحر، يرتكز على الورد الأسود والباريسيان فانيلا.
    4. Royal Keshmir: عطر النقاء والرفاهية، أخشاب كشميرية ناعمة مع المسك الأبيض.
    5. Royal Pisces: العطر المخصص المحدود (Limited Edition) المصنوع خصيصاً للدكتور مينا.
    """

# =========================================================
# [السجلات الملكية والسحابية - MongoDB]
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
        print(f"MongoDB Connection Error: {e}")

def save_to_db(phone, record_type, text, selfie=None, product=None):
    if not phone or vault_collection is None: return
    clean_phone = str(phone).strip()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if selfie:
        existing_selfies = list(vault_collection.find({"phone": {"$eq": clean_phone}, "selfie": {"$ne": None}}).sort("date", 1))
        if len(existing_selfies) >= 5:
            for i in range(len(existing_selfies) - 4):
                vault_collection.delete_one({"_id": existing_selfies[i]["_id"]})

    document = {"phone": clean_phone, "type": record_type, "text": text, "selfie": selfie, "product": product, "date": date_str}
    vault_collection.insert_one(document)

def get_history_from_db(phone, limit=10):
    if not phone or vault_collection is None: return ""
    clean_phone = str(phone).strip()
    records = list(vault_collection.find({"phone": {"$eq": clean_phone}}).sort("date", -1).limit(limit))
    if not records: return ""
    return " | ".join([f"[تاريخ {r['date']} - {r['type']}]: {r['text']}" for r in reversed(records)])

# =========================================================
# 🛡️ [درع الباقة المطور: عداد الصور لكل عميل]
# =========================================================
user_image_quota = {}
MAX_SIMULATIONS_PER_HOUR = 10

def check_and_update_quota(phone: str) -> bool:
    now = time.time()
    if phone not in user_image_quota:
        user_image_quota[phone] = []
    user_image_quota[phone] = [t for t in user_image_quota[phone] if now - t < 3600]
    if len(user_image_quota[phone]) >= MAX_SIMULATIONS_PER_HOUR:
        return False
    user_image_quota[phone].append(now)
    return True

# =========================================================
# [الوعي البصري والموديلات الرسمية لجوجل]
# =========================================================
TASK_FILE = 'face_landmarker.task'
TASK_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
if not os.path.exists(TASK_FILE): urllib.request.urlretrieve(TASK_URL, TASK_FILE)

base_options = python.BaseOptions(model_asset_path=TASK_FILE)
options = vision.FaceLandmarkerOptions(
    base_options=base_options, num_faces=1, min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5, min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

keys_string = os.environ.get("GOOGLE_API_KEYS", os.environ.get("GOOGLE_API_KEY", ""))
SYSTEM_API_KEYS = [key.strip() for key in keys_string.split(",") if key.strip() and key.strip().startswith("AIza")]

TEXT_MODELS = ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"]
VISION_MODELS = TEXT_MODELS.copy()

# =========================================================
# 🛠️ [محرك الرسوميات الذكي: فك التشفير والضغط والتحويل]
# =========================================================
def decode_b64_image(base64_string):
    try:
        if "," in base64_string: base64_string = base64_string.split(",")[1]
        base64_string += "=" * ((4 - len(base64_string) % 4) % 4)
        img_data = base64.b64decode(base64_string)
        return cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    except: return None

def resize_for_web(img_cv, max_dim=1024):
    if img_cv is None: return None
    h, w = img_cv.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img_cv, (int(w * scale), int(h * scale)))
    return img_cv

def extract_color_from_product(img_cv):
    try:
        h, w = img_cv.shape[:2]
        center_patch = img_cv[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        pixels = center_patch.reshape(-1, 3)
        valid_pixels = [p for p in pixels if not (np.all(p > 240) or np.all(p < 15))]
        if not valid_pixels: return (139, 0, 0)
        return tuple(map(int, np.mean(valid_pixels, axis=0)[::-1]))
    except: return (139, 0, 0)

def analyze_skin_tone_and_get_color(img_cv, makeup_type):
    try:
        image_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        res = face_landmarker.detect(mp_image)
        if not res.face_landmarks: return (180, 50, 50), "غير محدد"
        h, w = img_cv.shape[:2]; landmarks = res.face_landmarks[0]
        cheek = img_cv[int(landmarks[50].y * h), int(landmarks[50].x * w)]
        b, g, r = int(cheek[0]), int(cheek[1]), int(cheek[2])
        if r > g + 15: return (195, 85, 75), "دافئة (Warm)"
        elif b > g + 5: return (185, 55, 105), "باردة (Cool)"
        return (175, 70, 85), "حيادية (Neutral)"
    except: return (150, 50, 50), "غير محدد"

# =========================================================
# 💄 [محرك المكياج والنحت والجمال الشامل - 70%+ من المتطلبات]
# =========================================================
def apply_royal_makeup(image_cv: np.ndarray, color_rgb: tuple, makeup_type: str, texture: str = "matte"):
    try:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = face_landmarker.detect(mp_image)
        if not detection_result.face_landmarks: return image_cv, False

        height, width, _ = image_cv.shape
        face_landmarks = detection_result.face_landmarks[0]
        
        ZONES = {
            "lips": [[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185]],
            "eyeshadow": [[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]],
            "blush": [[116, 117, 118, 119, 100, 120, 121, 147, 213, 192, 214, 210, 211, 32, 208, 199], [345, 346, 347, 348, 329, 350, 351, 376, 433, 416, 434, 430, 431, 262, 428, 420]],
            "foundation": [[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]],
            "highlighter": [[116, 117, 118, 119, 147, 213], [345, 346, 347, 348, 376, 433], [197, 195, 5, 4]], 
            "concealer": [[226, 31, 228, 229, 230, 231, 232, 233, 244, 189], [446, 261, 448, 449, 450, 451, 452, 453, 464, 413]] 
        }
        
        target_zones = ZONES.get(makeup_type, ZONES["lips"])
        mask = np.zeros((height, width), dtype=np.uint8)
        for zone in target_zones:
            points = np.array([[int(face_landmarks[idx].x * width), int(face_landmarks[idx].y * height)] for idx in zone], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        
        blur_size = 35 if makeup_type in ["blush", "foundation", "concealer"] else 11
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        alpha = np.expand_dims(mask / 255.0, axis=-1)

        img_lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)

        color_layer_bgr = np.zeros_like(image_cv)
        color_layer_bgr[:] = color_rgb[::-1]
        color_lab = cv2.cvtColor(color_layer_bgr, cv2.COLOR_BGR2LAB)
        _, ca_channel, cb_channel = cv2.split(color_lab)

        if makeup_type == "foundation":
            smooth_skin = cv2.bilateralFilter(image_cv, 15, 75, 75)
            blended_layer = cv2.addWeighted(smooth_skin, 0.75, color_layer_bgr, 0.25, 0)
        elif makeup_type == "highlighter":
            blended_layer = cv2.add(image_cv, cv2.convertScaleAbs(color_layer_bgr, alpha=0.35))
        elif makeup_type == "concealer":
            bright_color = cv2.add(color_layer_bgr, (25, 25, 25))
            blended_layer = cv2.addWeighted(image_cv, 0.45, bright_color, 0.55, 0)
        else:
            strength = 0.75 if makeup_type in ["lips", "eyeshadow"] else 0.45
            new_a = cv2.addWeighted(a_channel, 1 - strength, ca_channel, strength, 0)
            new_b = cv2.addWeighted(b_channel, 1 - strength, cb_channel, strength, 0)
            merged_lab = cv2.merge((l_channel, new_a, new_b))
            blended_layer = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        if texture == "glossy" and makeup_type == "lips":
            _, highlights = cv2.threshold(l_channel, 175, 255, cv2.THRESH_BINARY)
            highlights_bgr = cv2.cvtColor(cv2.GaussianBlur(highlights, (5,5), 0), cv2.COLOR_GRAY2BGR)
            blended_layer = cv2.add(blended_layer, cv2.convertScaleAbs(highlights_bgr, alpha=0.45))

        final_image = (1.0 - alpha) * image_cv.astype(np.float32) + alpha * blended_layer.astype(np.float32)
        return final_image.astype(np.uint8), True
    except: return image_cv, False

# =========================================================
# 💇 [محرك الشعر والقصات الملكي الجديد - AI AR Haircuts]
# =========================================================
def apply_royal_haircut(image_cv: np.ndarray, haircut_style: str):
    """
    محرك تصميم الشعر والقصات: يحدد منطقة الرأس العلوية ويمنح العميل
    تأطيراً ضوئياً لونياً ذكياً ومحاكاة هيكلية للقصة المختارة.
    """
    try:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = face_landmarker.detect(mp_image)
        if not detection_result.face_landmarks: return image_cv, False

        height, width, _ = image_cv.shape
        face_landmarks = detection_result.face_landmarks[0]
        
        # تتبع النقاط العليا للجبهة لتحديد بداية منبت الشعر
        forehead_points = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
        xs = [int(face_landmarks[idx].x * width) for idx in forehead_points]
        ys = [int(face_landmarks[idx].y * height) for idx in forehead_points]
        
        min_x, max_x = max(0, min(xs) - 40), min(width, max(xs) + 40)
        min_y = max(0, min(ys) - 150) # الصعود لأعلى الرأس لمحاكاة الشعر
        max_y = max(ys)
        
        # إنشاء قناع الشعر الرسومي الملكي
        hair_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(hair_mask, (int((min_x + max_x)/2), int((min_y + max_y)/2)), 
                    (int((max_x - min_x)/2), int((max_y - min_y)/1.2)), 0, 0, 360, 255, -1)
        
        # إخفاء الوجه السفلي من قناع الشعر
        face_mask = np.zeros((height, width), dtype=np.uint8)
        face_pts = np.array([[int(face_landmarks[i].x * width), int(face_landmarks[i].y * height)] for i in range(150)], dtype=np.int32)
        cv2.fillPoly(face_mask, [face_pts], 255)
        hair_mask = cv2.subtract(hair_mask, face_mask)
        
        hair_mask = cv2.GaussianBlur(hair_mask, (45, 45), 0)
        alpha = np.expand_dims(hair_mask / 255.0, axis=-1)
        
        # تحديد الألوان والانعكاسات حسب نمط القصة
        style_colors = {
            "short_bob": (80, 40, 50),     # كستنائي فرنسي هادئ
            "long_waves": (130, 95, 65),   # عسلي ملكي متموج
            "pixie_cut": (40, 45, 60),     # أسود ليلكي جريء
            "curly_volume": (110, 70, 45)  # شوكولاتة كثيف متموج
        }
        chosen_color = style_colors.get(haircut_style, (100, 80, 70))
        
        color_layer = np.zeros_like(image_cv)
        color_layer[:] = chosen_color[::-1]
        
        # دمج ألوان وتفاصيل خصلات الشعر بذكاء
        blended = cv2.addWeighted(image_cv, 0.55, color_layer, 0.45, 0)
        final_image = (1.0 - alpha) * image_cv.astype(np.float32) + alpha * blended.astype(np.float32)
        
        return final_image.astype(np.uint8), True
    except:
        return image_cv, False

# =========================================================
# 🔄 [خوارزمية الدوران الذكية والمنيعة للأخطاء والمفاتيح]
# =========================================================
def robust_generate(client_api_key, contents, models_list):
    keys_to_use = [client_api_key.strip()] if client_api_key and client_api_key.strip() else SYSTEM_API_KEYS.copy()
    if not keys_to_use: raise HTTPException(status_code=500, detail="مفاتيح الخادم غير مهيأة.")
    
    safe_contents = contents[0] if isinstance(contents, list) and len(contents) == 1 else contents

    for model_name in models_list:
        random.shuffle(keys_to_use)
        model_failed = False
        for key in keys_to_use:
            if model_failed: break
            for attempt in range(2):
                try:
                    client = genai.Client(api_key=key)
                    config_tools = types.GenerateContentConfig(temperature=0.8, top_p=0.95, tools=[read_local_file])
                    config_no_tools = types.GenerateContentConfig(temperature=0.8, top_p=0.95)
                    
                    if isinstance(safe_contents, list):
                        response = client.models.generate_content(model=model_name, contents=safe_contents, config=config_no_tools)
                    else:
                        response = client.chats.create(model=model_name, config=config_tools).send_message(safe_contents)
                    
                    if response and response.text: return response.text
                except Exception as e:
                    err_str = str(e).lower()
                    print(f"Skipping key on [{model_name}]: {err_str[:70]}")
                    if "404" in err_str or "not found" in err_str or "is not supported" in err_str:
                        model_failed = True
                        break
                    elif "429" in err_str or "resourceexhausted" in err_str:
                        break
                    else:
                        time.sleep(1)
                        continue
                        
    raise HTTPException(status_code=503, detail="السيرفر ممتلئ، يرجى إعادة المحاولة.")

# =========================================================
# 🎯 [صمام أمان محرك الجرد: كشف وقراءة الملفات تلقائياً]
# =========================================================
def clean_qty_value(val):
    if val is None: return 0.0
    s = str(val).strip().replace(',', '.')
    if s == '' or s.lower() == 'nan': return 0.0
    try: return float(s)
    except: return 0.0

def get_qty_by_keyword(row, keywords):
    for col in row.keys():
        col_lower = str(col).lower()
        if any(x in col_lower for x in ['سعر', 'price', 'كود', 'code', 'باركود']): continue
        for kw in keywords:
            if kw in col_lower:
                val = clean_qty_value(row[col])
                if val >= 9999: return 0.0 # صمام أمان لإلغاء القيمة الوهمية 99999 في الغردقة
                return val
    return 0.0

def get_inventory() -> pd.DataFrame:
    """تكتشف ملف الجرد تلقائياً مهما كان رقمه بين قوسين"""
    try:
        files = os.listdir('.')
        target_file = next((f for f in files if f.endswith('.csv') and 'last' in f.lower()), None)
        if not target_file:
            target_file = next((f for f in files if f.endswith('.csv') and 'combined' in f.lower()), None)
        if target_file:
            return pd.read_csv(target_file, encoding='utf-8-sig', on_bad_lines='skip').fillna("")
        return pd.DataFrame()
    except: return pd.DataFrame()

def get_links_db() -> pd.DataFrame:
    try:
        if os.path.exists('royalelchim-app-2026-06-01-2.xlsx'):
            return pd.read_excel('royalelchim-app-2026-06-01-2.xlsx').fillna("")
        return pd.DataFrame()
    except: return pd.DataFrame()

# =========================================================
# [بوابات ونقاط الاتصال - API Endpoints]
# =========================================================
class ChatPayload(BaseModel): text: str; phone: str; client_api_key: Optional[str] = None
class PerfumePayload(BaseModel): client_message: str; phone: str; client_api_key: Optional[str] = None
class SimulationPayload(BaseModel): user_selfie: str; phone: str; product_image: Optional[str] = None; makeup_type: str = "lips"; texture: str = "matte"; client_api_key: Optional[str] = None
class HaircutPayload(BaseModel): user_selfie: str; phone: str; haircut_style: str = "long_waves"; client_api_key: Optional[str] = None

@app.get("/api/search")
async def search(query: str):
    try:
        inv = get_inventory()
        db_links = get_links_db()
        if inv.empty or 'الصنف' not in inv.columns:
            return {"status": "error", "message": "ملف البيانات غير متوفر حالياً على السيرفر."}

        query_words = query.strip().split()
        mask = pd.Series([True] * len(inv))
        for word in query_words:
            mask = mask & (inv['الصنف'].astype(str).str.contains(word, case=False, na=False) | inv['الباركود'].astype(str).str.contains(word, case=False, na=False))
        
        results = inv[mask].head(25)
        data = []
        for _, row in results.iterrows():
            item_name = str(row.get('الصنف', '')).strip()
            
            luxor = int(get_qty_by_keyword(row, ['اللوتس', 'لوتس', 'اقصر', 'luxor']))
            marwa = int(get_qty_by_keyword(row, ['المروة', 'المروه', 'marwa']))
            hurgada = int(get_qty_by_keyword(row, ['الغردقة', 'الغردقه', 'hurgada']))
            online = int(get_qty_by_keyword(row, ['اونلاين', 'online']))

            if any(kw in item_name.lower() for kw in ["زيت", "جرام", "تركيب", "كحول", "مثبت"]): luxor = 0
            price = clean_qty_value(row.get('سعر1 كارت', 0))
            
            product_url = ""
            if not db_links.empty and 'name' in db_links.columns:
                match = db_links[db_links['name'].astype(str).str.contains(item_name, case=False, regex=False, na=False)]
                if not match.empty and 'item_page_link' in db_links.columns:
                    product_url = str(match['item_page_link'].values[0]).strip()

            if not product_url:
                product_url = f"https://www.royalelchim.app/search?q={urllib.parse.quote(item_name)}"

            branches_text = f"🟢 المروة: {marwa} | 🔵 الغردقة: {hurgada} | 🟡 الأقصر: {luxor} | 🌐 الأونلاين: {online}"
            link_html = f"<br><a href='{product_url}' target='_blank' class='btn btn-sm btn-outline-warning mt-1' style='font-size:0.75rem;'><i class='fa-solid fa-cart-shopping'></i> شراء من الموقع</a>"
            full_name = f"<b>{item_name}</b> <br> <span class='text-muted small'>📦 ( {branches_text} )</span> {link_html}"
            
            data.append({"name": full_name, "price": price})
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/diagnose")
async def diagnose(payload: ChatPayload):
    history = get_history_from_db(payload.phone)
    prompt = f"{BASE_PHILOSOPHY}\nتاريخ العميل: [{history}]\nالطلب: '{payload.text}'"
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    save_to_db(payload.phone, "حديث الصداقة", f"الطلب: {payload.text}\nالرد: {res}")
    return {"status": "success", "diagnosis": res}

@app.post("/api/chat")
async def chat(payload: ChatPayload):
    history = get_history_from_db(payload.phone)
    prompt = f"{BASE_PHILOSOPHY}\nتاريخ العميل: [{history}]\nالطلب: '{payload.text}'"
    res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
    save_to_db(payload.phone, "استشارة", f"الطلب: {payload.text}\nالرد: {res}")
    return {"status": "success", "answer": res}

@app.post("/api/craft_perfume")
async def craft_perfume(payload: PerfumePayload):
    try:
        inv = get_inventory(); brand_catalog = get_brand_catalog(); available_oils = []
        if not inv.empty and 'الصنف' in inv.columns:
            oils_df = inv[inv['الصنف'].astype(str).str.contains("SAVVY|PARFUME OIL", case=False, na=False)]
            for _, row in oils_df.head(60).iterrows():
                price = clean_qty_value(row.get('سعر1 كارت', 0))
                if price > 0: available_oils.append(f"{str(row.get('الصنف','')).strip()} ({price} ج.م)")
        oils_list = " \n ".join(available_oils) if available_oils else "قيد التحديث."
        
        prompt = f"{BASE_PHILOSOPHY}\nتاريخ: [{get_history_from_db(payload.phone)}]\nطلب: '{payload.client_message}'\nالزيوت المتوفرة بالمخازن:\n{oils_list}\nالبراند الجاهز:\n{brand_catalog}\nصمم تركيبة بالجرامات دقيقة، احسب التكلفة، واربطها بقاعدة العطر الأساسي R.O.Y.A.L E.K.A بالنقط بين الحروف."
        res = robust_generate(payload.client_api_key, [prompt], TEXT_MODELS)
        save_to_db(payload.phone, "تصميم عطر عالي الدقة", f"الطلب: {payload.client_message}\nالرد: {res}")
        return {"status": "success", "answer": res}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate_makeup")
async def simulate_makeup(payload: SimulationPayload):
    if not check_and_update_quota(payload.phone):
        raise HTTPException(status_code=429, detail="لقد تجاوزتِ الحد الأقصى لتجارب الصور المجانية في هذه الساعة لحماية الخادم.")
    try:
        img_cv_original = decode_b64_image(payload.user_selfie)
        if img_cv_original is None: raise ValueError("الصورة مرتفعة الحجم أو تالفة.")

        rgb_color = (150, 50, 50); ai_context = ""
        if payload.product_image:
            prod_cv = decode_b64_image(payload.product_image)
            if prod_cv is not None:
                rgb_color = extract_color_from_product(resize_for_web(prod_cv, max_dim=400))
                ai_context = f"قمتِ برفع منتج، وقام محرك الرسوميات باستخراج اللون الحقيقي بدقة (RGB: {rgb_color}) وتطبيقه كـ {payload.makeup_type} بملمس {payload.texture}."
        else:
            rgb_color, skin_tone = analyze_skin_tone_and_get_color(img_cv_original, payload.makeup_type)
            ai_context = f"دقة الفحص البصري حللت بشرتكِ بأنها ({skin_tone}) وتم دمج الدرجة الفاخرة المناسبة لملامحك الملكية كـ {payload.makeup_type} بملمس {payload.texture}."

        processed_img_hd, face_found = apply_royal_makeup(img_cv_original, rgb_color, payload.makeup_type, payload.texture)
        
        if face_found:
            _, buffer_hd = cv2.imencode('.jpg', processed_img_hd, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            result_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_hd).decode('utf-8')}"
            small_cv = resize_for_web(processed_img_hd, max_dim=800)
            pil_image = Image.fromarray(cv2.cvtColor(small_cv, cv2.COLOR_BGR2RGB))
        else:
            result_base64 = payload.user_selfie
            small_cv = resize_for_web(img_cv_original, max_dim=800)
            pil_image = Image.fromarray(cv2.cvtColor(small_cv, cv2.COLOR_BGR2RGB))

        expert_prompt = f"{BASE_PHILOSOPHY}\n{ai_context}\nاشرحي للعميلة بأسلوب خبيرة مظهر ملكية كيف تلاءم هذا الملمس والدرجة ملامح وجهها الفريدة."
        res = robust_generate(payload.client_api_key, [pil_image, expert_prompt], VISION_MODELS)
        
        save_to_db(payload.phone, f"محاكاة {payload.makeup_type}", res, result_base64, payload.product_image)
        return {"status": "success", "result_image": result_base64, "simulation_result": res, "face_detected": face_found}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate_hair")
async def simulate_hair(payload: HaircutPayload):
    if not check_and_update_quota(payload.phone):
        raise HTTPException(status_code=429, detail="لقد تجاوزتِ حد محاكاة الصور المسموح به لهذه الساعة.")
    try:
        img_cv_original = decode_b64_image(payload.user_selfie)
        if img_cv_original is None: raise ValueError("صورة الوجه غير صالحة.")

        processed_img_hd, face_found = apply_royal_haircut(img_cv_original, payload.haircut_style)
        
        if face_found:
            _, buffer_hd = cv2.imencode('.jpg', processed_img_hd, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            result_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer_hd).decode('utf-8')}"
            small_cv = resize_for_web(processed_img_hd, max_dim=800)
            pil_image = Image.fromarray(cv2.cvtColor(small_cv, cv2.COLOR_BGR2RGB))
        else:
            result_base64 = payload.user_selfie
            small_cv = resize_for_web(img_cv_original, max_dim=800)
            pil_image = Image.fromarray(cv2.cvtColor(small_cv, cv2.COLOR_BGR2RGB))

        styles_map = {"short_bob": "باريزيان بوب الفرنسي الفرنسي قصير", "long_waves": "التموجات الملكية الكثيفة الطويلة", "pixie_cut": "البيكسي الجريء العصري", "curly_volume": "الكيرلي الإفريقي الكثيف"}
        chosen_style_ar = styles_map.get(payload.haircut_style, "ستايل ملكي مخصص")

        expert_prompt = f"{BASE_PHILOSOPHY}\nالعميلة قامت بمحاكاة قصة شعر: ({chosen_style_ar}). حللي شكل وجهها (بيضاوي، دافئ، إلخ) واشرحي لها بأسلوب مصفف شعر عالمي كيف تبرز هذه القصة مكامن جمال ملامحها."
        res = robust_generate(payload.client_api_key, [pil_image, expert_prompt], VISION_MODELS)
        
        save_to_db(payload.phone, f"قصة شعر {payload.haircut_style}", res, result_base64)
        return {"status": "success", "result_image": result_base64, "simulation_result": res, "face_detected": face_found}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vault")
async def get_vault(phone: str):
    if vault_collection is None: return {"status": "error"}
    records = list(vault_collection.find({"phone": {"$eq": str(phone).strip()}}).sort("date", -1))
    return {"status": "success", "data": [{"id": str(r['_id']), "type": r.get('type'), "text": r.get('text'), "selfie": r.get('selfie'), "product": r.get('product'), "date": r.get('date')} for r in records]}

@app.delete("/api/vault/{record_id}")
async def delete_vault_record(record_id: str, phone: str):
    if vault_collection is None: return {"status": "error"}
    try:
        res = vault_collection.delete_one({"_id": ObjectId(record_id), "phone": str(phone).strip()})
        return {"status": "success"} if res.deleted_count > 0 else {"status": "error"}
    except: return {"status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
