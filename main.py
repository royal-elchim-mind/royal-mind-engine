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

app = FastAPI(title="Royal Elchim - Omni-Conscious Enterprise")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👑 الشخصية الفلسفية والوجدانية لرويال مايند
BASE_PHILOSOPHY = (
    "أنتِ رويال مايند، العقل البرمجي والوجداني لبراند Royal Elchim العريق. "
    "تتحدثين كخبير تجميل وعطور وعناية بالشعر عالمي، بأسلوب فاخر، راقٍ، ومبهر للعميلة."
)

def get_brand_catalog() -> str:
    return """
    كتالوج براند Royal Elchim الفاخر:
    1. ROYAL E.K.A: العطر الأساسي الإمبراطوري الفخم (يجب كتابته بالنقط بين الحروف).
    2. Royal Ignite: عطر التوهج والشغف بنوتات العنبر والتوابل الفاخرة.
    3. Royal Midnight Rose: عطر الغموض الساحر بالورد الأسود والباريسيان فانيلا.
    4. Royal Keshmir: عطر النقاء والرفاهية بأخشاب الكشمير والمسك الأبيض.
    5. Royal Pisces: العطر المخصص المحدود المصنوع خصيصاً للدكتور مينا.
    """

# =========================================================
# [الأدوات التقنية - قراءة السجلات والملفات]
# =========================================================
def read_local_file(filename: str) -> str:
    """أداة للذكاء الاصطناعي لقراءة ملفات الجرد أو التركيبات من السيرفر"""
    try:
        files = os.listdir('.')
        target = next((f for f in files if filename.lower() in f.lower()), None)
        if not target: return f"الملف {filename} غير موجود."
        with open(target, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()[:20000]
    except Exception as e: return str(e)

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
# 🛡️ [درع حماية الباقة والمفاتيح المتعددة]
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
# [الوعي البصري ونماذج الرؤية]
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
# 💄 [محرك الرسوميات Vivid AR: دمج الميكب والنحت فائق التشبع]
# =========================================================
def apply_vivid_makeup(image_cv: np.ndarray, color_rgb: tuple, makeup_type: str, texture: str = "matte"):
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
            "blush": [[116, 117, 118, 119, 100, 120, 121, 147, 213, 192, 214, 210, 211], [345, 346, 347, 348, 329, 350, 351, 376, 433, 416, 434, 430, 431]],
            "foundation": [[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]],
            "highlighter": [[116, 117, 118, 119, 147, 213], [345, 346, 347, 348, 376, 433], [197, 195, 5, 4]],
            "concealer": [[226, 31, 228, 229, 230, 231, 232, 233], [446, 261, 448, 449, 450, 451, 452, 453]]
        }
        
        target_zones = ZONES.get(makeup_type, ZONES["lips"])
        mask = np.zeros((height, width), dtype=np.uint8)
        for zone in target_zones:
            points = np.array([[int(face_landmarks[idx].x * width), int(face_landmarks[idx].y * height)] for idx in zone], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        
        blur_radius = 11 if makeup_type in ["lips", "eyeshadow"] else 45
        mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
        alpha = np.expand_dims(mask / 255.0, axis=-1)
        
        color_layer = np.zeros_like(image_cv)
        color_layer[:] = color_rgb[::-1]
        
        if makeup_type == "foundation":
            smooth_skin = cv2.bilateralFilter(image_cv, 15, 75, 75)
            blended = cv2.addWeighted(smooth_skin, 0.75, color_layer, 0.25, 0)
        elif makeup_type == "highlighter":
            shimmer = np.zeros_like(image_cv)
            shimmer[:] = (210, 235, 255)
            blended = cv2.addWeighted(image_cv, 0.65, shimmer, 0.35, 0)
        elif makeup_type == "concealer":
            smooth_eye = cv2.bilateralFilter(image_cv, 9, 50, 50)
            brighten = np.zeros_like(image_cv)
            brighten[:] = (185, 215, 245)
            blended = cv2.addWeighted(smooth_eye, 0.55, brighten, 0.45, 0)
        else:
            # 🎯 دمج فائق الوضوح ومشبّع للألوان (Lips / Eyeshadow / Blush) بنسبة ظلام 85%
            str_val = 0.85 if makeup_type in ["lips", "eyeshadow"] else 0.5
            blended = cv2.addWeighted(image_cv, 1 - str_val, color_layer, str_val, 0)
            
            if texture == "glossy" and makeup_type == "lips":
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                _, high = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
                high_mask = cv2.bitwise_and(high, mask)
                high_blur = cv2.GaussianBlur(high_mask, (5, 5), 0)
                blended = cv2.add(blended, cv2.cvtColor(high_blur, cv2.COLOR_GRAY2BGR))

        final_image = (1.0 - alpha) * image_cv.astype(np.float32) + alpha * blended.astype(np.float32)
        return final_image.astype(np.uint8), True
    except:
        return image_cv, False

# =========================================================
# 💇 [محرك هندسة النسيج وتصميم قصات الشعر - AI Haircuts]
# =========================================================
def apply_royal_haircuts_engine(image_cv: np.ndarray, style: str):
    try:
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        res = face_landmarker.detect(mp_image)
        if not res.face_landmarks: return image_cv, False
        h, w, _ = image_cv.shape
        lms = res.face_landmarks[0]
        
        pt10 = (int(lms[10].x * w), int(lms[10].y * h))
        pt152 = (int(lms[152].x * w), int(lms[152].y * h))
        face_height = pt152[1] - pt10[1]
        
        hair_mask = np.zeros((h, w), dtype=np.uint8)
        hair_color = (25, 30, 40) # لون بني داكن فاخر
        
        if style == "short_bob":
            pts = [
                (int(lms[21].x * w) - 45, pt10[1] - int(face_height * 0.25)),
                (pt10[0], pt10[1] - int(face_height * 0.35)),
                (int(lms[251].x * w) + 45, pt10[1] - int(face_height * 0.25)),
                (int(lms[454].x * w) + 40, int(lms[454].y * h) + 30),
                (int(lms[288].x * w), int(lms[152].y * h) - 5),
                (int(lms[58].x * w), int(lms[152].y * h) - 5),
                (int(lms[234].x * w) - 40, int(lms[234].y * h) + 30),
            ]
            cv2.fillPoly(hair_mask, [np.array(pts, np.int32)], 255)
            bangs_bottom = int((lms[105].y + lms[334].y) * 0.5 * h)
            cv2.rectangle(hair_mask, (int(lms[21].x * w), bangs_bottom), (int(lms[251].x * w), h), 0, -1)
            
        elif style == "long_waves":
            pts = [
                (int(lms[21].x * w) - 55, pt10[1] - int(face_height * 0.3)),
                (pt10[0], pt10[1] - int(face_height * 0.4)),
                (int(lms[251].x * w) + 55, pt10[1] - int(face_height * 0.3)),
                (int(lms[454].x * w) + 65, int(lms[454].y * h)),
                (w, h), (int(lms[361].x * w) + 35, h),
                (int(lms[454].x * w) + 15, int(lms[152].y * h)),
                (pt152[0], pt152[1] + 25),
                (int(lms[234].x * w) - 15, int(lms[152].y * h)),
                (int(lms[132].x * w) - 35, h), (0, h),
                (int(lms[234].x * w) - 65, int(lms[234].y * h)),
            ]
            cv2.fillPoly(hair_mask, [np.array(pts, np.int32)], 255)
            
        elif style == "pixie_cut":
            pts = [
                (int(lms[21].x * w) - 25, pt10[1] - int(face_height * 0.15)),
                (pt10[0], pt10[1] - int(face_height * 0.3)),
                (int(lms[251].x * w) + 25, pt10[1] - int(face_height * 0.15)),
                (int(lms[454].x * w) + 15, int(lms[251].y * h)),
                (int(lms[251].x * w) - 15, pt10[1] + 15),
                (int(lms[21].x * w) + 15, pt10[1] + 15),
                (int(lms[234].x * w) - 15, int(lms[21].y * h)),
            ]
            cv2.fillPoly(hair_mask, [np.array(pts, np.int32)], 255)
            
        else: # curly_volume
            cv2.ellipse(hair_mask, (pt10[0], pt10[1] - 15), (int(w * 0.48), int(face_height * 0.55)), 0, 0, 360, 255, -1)
            cv2.ellipse(hair_mask, (int(lms[21].x * w), int(lms[21].y * h)), (int(w * 0.22), int(h * 0.28)), 0, 0, 360, 255, -1)
            cv2.ellipse(hair_mask, (int(lms[251].x * w), int(lms[251].y * h)), (int(w * 0.22), int(h * 0.28)), 0, 0, 360, 255, -1)

        face_clear_mask = np.zeros((h, w), dtype=np.uint8)
        face_features_nodes = [162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]
        face_pts = np.array([[int(lms[idx].x * w), int(lms[idx].y * h)] for idx in face_features_nodes], np.int32)
        cv2.fillPoly(face_clear_mask, [face_pts], 255)
        
        if style == "short_bob":
            eyebrow_level = int((lms[105].y + lms[334].y) * 0.5 * h)
            cv2.rectangle(face_clear_mask, (0, 0), (w, eyebrow_level), 0, -1)

        hair_mask = cv2.subtract(hair_mask, face_clear_mask)
        hair_mask_blurred = cv2.GaussianBlur(hair_mask, (7, 7), 0)
        alpha = np.expand_dims(hair_mask_blurred / 255.0, axis=-1)
        
        hair_layer = np.zeros_like(image_cv)
        hair_layer[:] = hair_color[::-1]
        
        # 🎯 رسم خطوط ونسيج نسيجي تفصيلي يحاكي خصلات الشعر في الواقع المعزز
        for i in range(0, h, 4):
            for j in range(0, w, 4):
                if hair_mask[i, j] > 0:
                    strand_brightness = random.randint(-20, 25)
                    cv2.line(hair_layer, (j, i), (j + random.choice([-2, 0, 2]), i + 4), 
                             (max(0, min(255, hair_color[0] + strand_brightness)),
                              max(0, min(255, hair_color[1] + strand_brightness)),
                              max(0, min(255, hair_color[2] + strand_brightness))), 1)

        final_image = (1.0 - alpha) * image_cv.astype(np.float32) + alpha * hair_layer.astype(np.float32)
        return final_image.astype(np.uint8), True
    except:
        return image_cv, False

# =========================================================
# [محرك التوصيل التلقائي وبيانات الجرد المركزية]
# =========================================================
def decode_b64_image(base64_string):
    try:
        if "," in base64_string: base64_string = base64_string.split(",")[1]
        base64_string += "=" * ((4 - len(base64_string) % 4) % 4)
        return cv2.imdecode(np.frombuffer(base64.b64decode(base64_string), np.uint8), cv2.IMREAD_COLOR)
    except: return None

def resize_for_web(img_cv, max_dim=1024):
    if img_cv is None: return None
    h, w = img_cv.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img_cv, (int(w * scale), int(h * scale)))
    return img_cv

def get_inventory() -> pd.DataFrame:
    """صمام أمان ديناميكي يكتشف أي نسخة مرفوعة من ملف الجرد تلقائياً"""
    try:
        files = os.listdir('.')
        target = next((f for f in files if f.endswith('.csv') and 'last' in f.lower()), None)
        if target: return pd.read_csv(target, encoding='utf-8-sig', on_bad_lines='skip').fillna("")
        return pd.DataFrame()
    except: return pd.DataFrame()

def get_links_db() -> pd.DataFrame:
    try:
        target = 'royalelchim-app-2026-06-01-2.xlsx'
        if os.path.exists(target): return pd.read_excel(target).fillna("")
        return pd.DataFrame()
    except: return pd.DataFrame()

def robust_generate(contents, models=TEXT_MODELS):
    keys = SYSTEM_API_KEYS.copy()
    random.shuffle(keys)
    for model in models:
        for key in keys:
            try:
                client = genai.Client(api_key=key)
                config = types.GenerateContentConfig(tools=[read_local_file], temperature=0.7)
                res = client.models.generate_content(model=model, contents=contents, config=config)
                if res and res.text: return res.text
            except: continue
    raise HTTPException(status_code=503, detail="خوادم الذكاء الاصطناعي ممتلئة حالياً.")

# =========================================================
# [بوابات ونقاط الاتصال المربوطة - API Routes]
# =========================================================
@app.get("/api/search")
async def search(query: str):
    try:
        inv = get_inventory()
        db_links = get_links_db()
        if inv.empty or 'الصنف' not in inv.columns:
            return {"status": "error", "message": "جدول البيانات غير متوفر على السيرفر."}

        query_words = query.strip().split()
        mask = pd.Series([True] * len(inv))
        for word in query_words:
            mask = mask & (inv['الصنف'].astype(str).str.contains(word, case=False, na=False) | inv['الباركود'].astype(str).str.contains(word, case=False, na=False))
        
        results = inv[mask].head(20)
        data = []
        for _, row in results.iterrows():
            item_name = str(row.get('الصنف', '')).strip()
            price = row.get('سعر1 كارت', 0)
            
            def clean_qty(k):
                val = row.get(k, 0)
                try: 
                    clean_v = float(str(val).replace(',','.'))
                    return 0 if clean_v >= 9999 else int(clean_v)
                except: return 0
                
            online = clean_qty('رويال الكيم اونلاين')
            luxor = clean_qty('رويال الكيم / سنتر اللوتس التجاري')
            hurgada = clean_qty('ROYAL ELCHIM . HURGADA')
            marwa = clean_qty('ROYAL ELCHIM MARWA')
            
            product_url = ""
            if not db_links.empty and 'name' in db_links.columns:
                match = db_links[db_links['name'].astype(str).str.contains(item_name, case=False, regex=False, na=False)]
                if not match.empty and 'item_page_link' in db_links.columns:
                    product_url = str(match['item_page_link'].values[0]).strip()

            if not product_url:
                product_url = f"https://www.royalelchim.app/search?q={urllib.parse.quote(item_name)}"

            branches_html = f"🌐 الأونلاين: {online} | 🔵 الغردقة: {hurgada} | 🟢 المروة: {marwa} | 🟡 الأقصر: {luxor}"
            full_html = f"<b>{item_name}</b><br><span class='text-muted small'>📦 ( {branches_html} )</span><br><a href='{product_url}' target='_blank' class='btn btn-sm btn-outline-warning mt-1'><i class='fa-solid fa-cart-shopping'></i> شراء من الموقع</a>"
            data.append({"name": full_html, "price": price})
            
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class SimulationPayload(BaseModel): user_selfie: str; phone: str; product_image: Optional[str] = None; makeup_type: str = "lips"; texture: str = "matte"; style: str = "long_waves"

@app.post("/api/simulate_makeup")
async def simulate_makeup(payload: SimulationPayload):
    if not check_and_update_quota(payload.phone):
        raise HTTPException(status_code=429, detail="استنفدتِ حصة المحاكاة لهاتفكِ هذه الساعة.")
    try:
        img_cv = decode_b64_image(payload.user_selfie)
        if img_cv is None: raise ValueError("السيلفي تالف.")
        img_cv = cv2.resize(img_cv, (800, int(800 * img_cv.shape[0] / img_cv.shape[1])))
        
        color = (145, 45, 55)
        if payload.product_image:
            p_img = decode_b64_image(payload.product_image)
            if p_img is not None:
                h, w = p_img.shape[:2]
                color = tuple(map(int, np.mean(p_img[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)], axis=(0,1))[::-1]))

        processed, face_found = apply_vivid_makeup(img_cv, color, payload.makeup_type, payload.texture)
        _, buf = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        res_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
        
        pil_img = Image.open(io.BytesIO(buf))
        prompt = f"{BASE_PHILOSOPHY}\nالعميلة تطبق {payload.makeup_type} بملمس {payload.texture}. اشرحي تناسق الإطلالة مع ملامحها."
        ai_res = robust_generate([pil_img, prompt])
        
        save_to_db(payload.phone, f"محاكاة {payload.makeup_type}", ai_res, res_b64)
        return {"status": "success", "result_image": res_b64, "simulation_result": ai_res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate_hair")
async def simulate_hair(payload: SimulationPayload):
    if not check_and_update_quota(payload.phone):
        raise HTTPException(status_code=429, detail="استنفدتِ حصة تجارب الشعر لهذه الساعة.")
    try:
        img_cv = decode_b64_image(payload.user_selfie)
        if img_cv is None: raise ValueError("السيلفي تالف.")
        
        processed, face_found = apply_royal_haircuts_engine(img_cv, payload.style)
        _, buf = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        res_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
        
        pil_img = Image.open(io.BytesIO(buf))
        prompt = f"{BASE_PHILOSOPHY}\nالعميلة تجرب قصة شعر {payload.style}. حللي تماشيه مع زوايا وجهها الفخم."
        ai_res = robust_generate([pil_img, prompt])
        
        save_to_db(payload.phone, "قصة شعر معززة", ai_res, res_b64)
        return {"status": "success", "result_image": res_b64, "simulation_result": ai_res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(payload: Dict):
    hist = get_history_from_db(payload.get('phone'))
    res = robust_generate([f"{BASE_PHILOSOPHY}\nتاريخ العميل السلوكي: {hist}\nالطلب الراهن: {payload.get('text')}"])
    save_to_db(payload.get('phone'), "استشارة ذكية", res)
    return {"status": "success", "answer": res}

@app.post("/api/craft_perfume")
async def craft_perfume(payload: Dict):
    catalog = get_brand_catalog()
    res = robust_generate([f"{BASE_PHILOSOPHY}\nالكتالوج المتاح:\n{catalog}\nصمم تركيبة عطرية مذهلة بالجرامات مع حساب دقيق ومقترح براند جاهز بناءً على: {payload.get('client_message')}، وتذكر كتابة العطر الإمبراطوري R.O.Y.A.L E.K.A بالنقط بين الحروف."])
    save_to_db(payload.get('phone'), "تصميم عطر ملكي", res)
    return {"status": "success", "answer": res}

@app.get("/api/vault")
async def vault(phone: str):
    if not vault_collection: return {"status": "error"}
    recs = list(vault_collection.find({"phone": str(phone).strip()}).sort("date", -1))
    return {"status": "success", "data": [{"id": str(r['_id']), "type": r['type'], "text": r['text'], "selfie": r.get('selfie'), "date": r['date']} for r in recs]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
