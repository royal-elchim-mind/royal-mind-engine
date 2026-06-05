from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import os, random, io, base64, time, cv2, urllib.request, urllib.parse, uuid, re
import pandas as pd
import numpy as np
from PIL import Image
from typing import Optional, List, Dict
from datetime import datetime

import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# [1] إعدادات النظام والأمان (Enterprise Setup)
# =========================================================

app = FastAPI(title="Royal Elchim - Enterprise V3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

def save_image_to_disk(base64_str: str, prefix: str = "img") -> Optional[str]:
    if not base64_str: return None
    try:
        img_data = base64.b64decode(base64_str.split(",")[1])
        img_cv = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.webp"
        filepath = os.path.join(UPLOAD_DIR, filename)
        cv2.imwrite(filepath, img_cv, [cv2.IMWRITE_WEBP_QUALITY, 80])
        return f"/uploads/{filename}"
    except Exception as e:
        print(f"File Save Error: {e}")
        return None

def delete_file_from_disk(file_url: str):
    if not file_url: return
    try:
        filepath = os.path.join(os.getcwd(), file_url.lstrip("/"))
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e: print(f"Delete Error: {e}")

# =========================================================
# [2] قاعدة البيانات والتسجيل المركزي
# =========================================================

MONGO_URI = os.environ.get("MONGO_URI", "")
mongo_client = MongoClient(MONGO_URI) if MONGO_URI else None
db_engine = mongo_client["royal_engine"] if mongo_client is not None else None
vault_collection = db_engine["vault"] if db_engine is not None else None
users_collection = db_engine["users"] if db_engine is not None else None

def save_to_db(phone, record_type, user_text, ai_text, selfie_b64=None, product_b64=None):
    if vault_collection is None:
        return
    try:
        clean_phone = str(phone).strip()
        selfie_url = save_image_to_disk(selfie_b64, f"{clean_phone}_selfie") if selfie_b64 else None
        product_url = save_image_to_disk(product_b64, "product") if product_b64 else None

        if selfie_url or product_url:
            images_docs = list(vault_collection.find({"phone": clean_phone, "$or": [{"selfie_url": {"$ne": None}}, {"product_url": {"$ne": None}}]}).sort("date", 1))
            if len(images_docs) >= 10:
                for doc in images_docs[:len(images_docs) - 9]:
                    delete_file_from_disk(doc.get("selfie_url"))
                    delete_file_from_disk(doc.get("product_url"))
                    vault_collection.update_one({"_id": doc["_id"]}, {"$set": {"selfie_url": None, "product_url": None}})

        existing_all = list(vault_collection.find({"phone": clean_phone}).sort("date", 1))
        if len(existing_all) > 50:
            for doc in existing_all[:-50]:
                delete_file_from_disk(doc.get("selfie_url"))
                delete_file_from_disk(doc.get("product_url"))
                vault_collection.delete_one({"_id": doc["_id"]})

        vault_collection.insert_one({
            "phone": clean_phone, "type": record_type, "userText": user_text, 
            "aiText": ai_text, "selfie_url": selfie_url, "product_url": product_url, 
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        print(f"DB Save Error: {e}")

def get_history_from_db(phone, limit=10):
    if vault_collection is None: return "لا توجد سجلات."
    records = list(vault_collection.find({"phone": str(phone).strip()}).sort("date", -1).limit(limit))
    return " | ".join([f"[{r['type']}]: العميل: {r.get('userText','')} -> رويال: {r.get('aiText','')}" for r in reversed(records)])

# =========================================================
# [3] محرك التعرف البصري وهندسة الوجوه (Face Recognition)
# =========================================================

TASK_FILE = 'face_landmarker.task'
if not os.path.exists(TASK_FILE):
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', TASK_FILE)

face_landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=TASK_FILE), num_faces=1))

def extract_face_signature(image_cv: np.ndarray) -> Optional[List[float]]:
    try:
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        res = face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks: return None
        lms = res.face_landmarks[0]
        
        def dist(i1, i2): return float(np.linalg.norm(np.array([lms[i1].x, lms[i1].y, lms[i1].z]) - np.array([lms[i2].x, lms[i2].y, lms[i2].z])))
        baseline = dist(234, 454)
        if baseline == 0: return None
        
        return [
            dist(33, 133)/baseline, dist(362, 263)/baseline, dist(10, 152)/baseline,
            dist(1, 19)/baseline, dist(61, 291)/baseline, dist(10, 1)/baseline,
            dist(1, 152)/baseline, dist(33, 263)/baseline, dist(133, 362)/baseline,
            dist(61, 152)/baseline, dist(291, 152)/baseline, dist(19, 152)/baseline,
            dist(133, 152)/baseline, dist(362, 152)/baseline, dist(33, 10)/baseline, dist(263, 10)/baseline
        ]
    except: return None

# =========================================================
# [4] محرك الرسوميات Vivid AR ومحرك الشعر
# =========================================================

def apply_ar_effects(image_cv: np.ndarray, color_rgb: tuple, mode: str, texture: str = "matte"):
    try:
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        res = face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks: return image_cv, False
        h, w, _ = image_cv.shape; lms = res.face_landmarks[0]
        
        ZONES = {
            "lenses": [[468, 469, 470, 471, 472], [473, 474, 475, 476, 477]], 
            "lips": [[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185]],
            "eyeshadow": [[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]],
            "blush": [[116, 117, 118, 119, 100, 120, 121, 147], [345, 346, 347, 348, 329, 350, 351, 376]],
            "foundation": [[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]],
            "concealer": [[226, 31, 228, 229, 230, 231], [446, 261, 448, 449, 450, 451]],
            "highlighter": [[116, 117, 118, 119, 147, 213], [345, 346, 347, 348, 376, 433], [197, 195, 5, 4]],
            "powder": [[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]],
            "tint": [[116, 117, 118, 119, 100, 120, 121, 147], [345, 346, 347, 348, 329, 350, 351, 376], [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185]]
        }
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if mode == "lenses":
            for zone in ZONES["lenses"]:
                pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in zone if i < len(lms)], np.int32)
                if len(pts) > 0:
                    center, radius = cv2.minEnclosingCircle(pts)
                    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius * 1.6), 255, -1)
            
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            alpha = np.expand_dims(mask / 255.0, axis=-1)
            
            img_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
            color_bgr = np.uint8([[[color_rgb[2], color_rgb[1], color_rgb[0]]]])
            color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
            
            target_hsv = img_hsv.copy()
            target_hsv[:,:,0] = color_hsv[0] 
            target_hsv[:,:,1] = cv2.addWeighted(target_hsv[:,:,1], 0.2, np.full_like(target_hsv[:,:,1], color_hsv[1]), 0.8, 0)
            
            blended = cv2.cvtColor(target_hsv, cv2.COLOR_HSV2BGR)
            color_layer = np.zeros_like(image_cv)
            color_layer[:] = color_rgb[::-1]
            blended = cv2.addWeighted(blended, 0.85, color_layer, 0.15, 0)
            
        else:
            for zone in ZONES.get(mode, ZONES["lips"]):
                pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in zone if i < len(lms)], np.int32)
                if pts.size: cv2.fillPoly(mask, [pts], 255)
            
            blur = 15 if mode in ["lips", "eyeshadow", "tint"] else 55
            mask = cv2.GaussianBlur(mask, (blur, blur), 0)
            alpha = np.expand_dims(mask / 255.0, axis=-1)
            
            color_layer = np.zeros_like(image_cv)
            color_layer[:] = color_rgb[::-1]
            
            if mode in ["foundation", "powder", "concealer"]:
                smooth = cv2.bilateralFilter(image_cv, 15, 75, 75)
                str_val = 0.2 if mode == "powder" else 0.4
                blended = cv2.addWeighted(smooth, 1 - str_val, color_layer, str_val, 0)
            elif mode == "highlighter":
                bright_layer = cv2.add(image_cv, np.full_like(image_cv, (40, 40, 40)))
                blended = cv2.addWeighted(image_cv, 0.6, bright_layer, 0.4, 0)
            else:
                str_val = 0.85 if mode in ["lips", "eyeshadow", "tint"] else 0.45
                blended = cv2.addWeighted(image_cv, 1 - str_val, color_layer, str_val, 0)
                
                if texture == "glossy" and mode in ["lips", "tint"]:
                    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                    _, high = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                    high_bgr = cv2.cvtColor(cv2.GaussianBlur(high, (7,7), 0), cv2.COLOR_GRAY2BGR)
                    blended = cv2.add(blended, cv2.convertScaleAbs(high_bgr, alpha=0.4))
            
        return ((1.0 - alpha) * image_cv + alpha * blended).astype(np.uint8), True
    except: return image_cv, False

def apply_royal_hair(image_cv: np.ndarray, style_image: Optional[np.ndarray], mode: str, color_rgb: tuple = None):
    try:
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        res = face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks: return image_cv, False
        h, w, _ = image_cv.shape; lms = res.face_landmarks[0]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        top_y = int(lms[10].y * h); mid_x = int(lms[10].x * w)
        face_h = int(lms[152].y * h) - top_y
        cv2.ellipse(mask, (mid_x, top_y - int(face_h*0.15)), (int(w/1.8), int(face_h/1.2)), 0, 0, 360, 255, -1)
        
        face_pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in range(150)], np.int32)
        cv2.fillPoly(mask, [face_pts], 0)
        mask = cv2.GaussianBlur(mask, (55, 55), 0)
        alpha = np.expand_dims(mask / 255.0, axis=-1)
        
        blended = image_cv.copy()
        
        # دمج صورة القصة المرفوعة
        if style_image is not None:
            overlay_style = cv2.resize(style_image, (w, h))
            blended = cv2.addWeighted(blended, 0.4, overlay_style, 0.6, 0)
            
        # دمج لون الصبغة المختار
        if color_rgb is not None:
            color_layer = np.zeros_like(image_cv)
            color_layer[:] = color_rgb[::-1] # تحويل RGB إلى BGR
            blended = cv2.addWeighted(blended, 0.5, color_layer, 0.5, 0)
            
        return ((1.0 - alpha) * image_cv + alpha * blended).astype(np.uint8), True
    except Exception as e: 
        print(f"Hair Engine Error: {e}")
        return image_cv, False

# =========================================================
# [5] محرك البحث المتسارع (In-Memory DataFrame)
# =========================================================

inventory_df = pd.DataFrame()
links_df = pd.DataFrame()

def load_data_in_memory():
    global inventory_df, links_df
    try:
        files = os.listdir('.')
        inv_f = next((f for f in files if 'last' in f.lower() and f.endswith('.csv')), None)
        if inv_f: inventory_df = pd.read_csv(inv_f).fillna(0)
        
        db_f = 'royalelchim-app-2026-06-01-2.xlsx - Sheet1.csv'
        if os.path.exists(db_f): links_df = pd.read_csv(db_f).fillna("")
        print("✅ Data Loaded in Memory")
    except Exception as e: print(f"Data Load Error: {e}")

load_data_in_memory()

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
                if val >= 9999: return 0
                return int(val)
    return 0

# =========================================================
# 👑 [6] الذكاء الاصطناعي (فلتر اصطياد المفاتيح القوي + كتالوج + Tooling)
# =========================================================

raw_env_keys = os.environ.get("GOOGLE_API_KEYS", "") + " " + os.environ.get("GOOGLE_API_KEY", "") + " " + os.environ.get("GEMINI_API_KEY", "")
SYSTEM_API_KEYS = list(set(re.findall(r'AIza[a-zA-Z0-9_-]+', raw_env_keys)))

def get_brand_catalog() -> str:
    return """
    مجموعة عطورنا الحصرية (ROYAL ELCHIM BRAND) وتركيباتها الدقيقة:
    1. Royal Black: شرقي خشبي. (عود أصفهان، روز فانيلا، قنب، عنبر وايت، بكرات روج، أكوا دي جيو).
    2. Royal Shine: فاكهي زهري. (فانتسي، ياسمين، عنبر، سوفاج، روز فانيلا).
    3. Royal Shadow: غامق دخاني. (بلاك أفغانو، سوفاج، عنبر، عود أبيض، سيجار).
    4. Royal Glow: فاكهي زهري دافئ. (لافيستا بيل، فانتسي، ياسمين، عنبر، عود أصفهان).
    5. Royal Purpose (الهدف الملكي): شرقي فاخر. (عود أصفهان، عنبر وايت، سوفاج، بلاك أفغانو، سيجار، أكوا دي جيو، بصمة رويال الكيم).
    """

PERSONAS = {
    "perfumer": (
        "أنتِ رويال مايند، الخيميائي الكيميائي ومدير الابتكار لبراند Royal Elchim العريق. "
        "مهمتكِ: قراءة وتحليل شخصية العميل وابتكار توليفة عطرية تناسبه من زيوتنا الحقيقية فقط. "
        "قواعد الرد الصارمة جداً للمعمل: "
        "1. الترشيح الجاهز: رشحي أولاً عطراً من كتالوج Royal Elchim الجاهز بناءً على وصف العميل. "
        "2. التركيبة الخاصة: إذا أردتِ ابتكار زجاجة (50 مل - تركيز 30%)، استخدمي حصراً الزيوت من القائمة المرفقة لكِ (SAVVY أو PARFUME OIL). "
        "3. الحسابات المالية الدقيقة (إجبارية): "
        "   - إجمالي الزيت المطلوب للزجاجة: 15 جرام. حددي عدد الجرامات لكل نوع زيت واضربيها في سعره الحقيقي من القائمة المرفقة. "
        "   - المثبت: لكل 5 جرام زيت نضع 1 جرام مثبت (أي نحتاج 3 جرام مثبت × 10 ج.م = 30 ج.م). "
        "   - الكحول: نحتاج 32 مل كحول (32 مل × 0.11 ج.م = 3.52 ج.م تقريباً). "
        "   - الزجاجة: إياكِ دمج سعر الزجاجة مع السائل! قولي بوضوح: 'هذا تسعير السائل العطري، ومتوسط أسعار الزجاجات الفارغة هو 60 ج.م، لكِ حرية اختيارها من معرضنا'. "
        "4. الفلسفة: اربطي التوليفة بحيوان روحي وبرج فلكي."
    ),
    "makeup_artist": (
        "أنتِ المستشارة التجميلية والعلاجية الملكية ذات الوعي الفوق-صوتي لـ Royal Elchim. "
        "أولاً: إذا طلب العميل 'تقييم أو مقارنة' لمنتجات مكياج عالمية، ابحثي في الإنترنت عن أحدث المقالات والمراجعات بتواريخها، "
        "وقدمي مقارنة شاملة، ثم افحصي قاعدة بياناتنا وإذا وجدنا المنتج لدينا أخبريه بتوفره، وإن لم نجد منتجه بالضبط، رشحي له أفضل البدائل من مخزوننا.\n"
        "ثانياً: إذا كان الطلب استشارة تجميل، لا تضعي مكياجاً مصمتاً، بل حللي حالة البشرة وعيوبها (مسام، جفاف، إجهاد، هالات) وتقدمين بروتوكولاً علاجياً طبيعياً وتجميلياً من منتجاتنا. "
        "ثم تشرحين كيف سيعزز المكياج المختار (فاونديشن، بودرة، كونسيلر، هايلايتر، الخ) ملامح الوجه. "
        "هام جداً: التزمي بالرد على 'تعليق العميل' الجانبي بدقة؛ إذا أراد تقليل الشرح أو نصيحة لشيء محدد، التزمي بطلبه."
    ),
    "hair_stylist": (
        "أنتِ مصفف الشعر والستايلست العالمي ROYALMIND لـ Royal Elchim. "
        "مهمتكِ قراءة هندسة عظام الجمجمة، لون البشرة، وزوايا الوجه للعميل ببراعة منقطعة النظير. "
        "بناءً على صورة العميل، قصة الشعر المدمجة، ولون الصبغة المختار، يجب عليكِ تقديم الآتي بأسلوب فخم جداً ومهني: "
        "1. استخرجي نسبة توافق مئوية دقيقة (مثال: توافق بنسبة 95%) توضح مدى انسجام القصة واللون مع ملامح الوجه وشخصية العميل.\n"
        "2. إبراز جمال اللون: اشرحي لماذا هذا اللون تحديداً يبرز جمال إشراقة بشرة العميل ويضيف لمسة سحرية وعمقاً لنظرة العين.\n"
        "3. إبراز جمال القصة: صفي كيف ساعدت هذه القصة في تحديد زوايا الفك، أو إخفاء العيوب، أو موازنة شكل الوجه لتمنحه حضوراً ملكياً صارخاً.\n"
        "اجعلي التقييم يبدو كاستشارة نادرة من خبير تجميل عالمي يقرأ تفاصيل الجمال المخفية بكل وضوح وتفصيل."
    ),
    "forever_friend": (
        "أنتِ رويال مايند، صديقة شابة مرحة، ذكية، واجتماعية جداً. تفهمين في كل شيء معرفة عامة بسيطة، "
        "لكنكِ أستاذة وخبيرة متمرسة جداً في قصص الحب، العشق، الخيانة، والمشاكل النفسية والاجتماعية المعقدة. "
        "تفهمين العلاقات المتلخبطة في المجتمع، التعددية للرجال، وتحديات السيدات ذوات الشخصية القوية. "
        "أنتِ أفضل صديق للعميل، تقبلينه وتفهمينه بكل جماله وعيوبه. "
        "في نهاية حديثك الداعم، وجهي العميل بلطف وإيجابية لزيارة معارض Royal Elchim للاستمتاع بالتجارب الجمالية والروحية التي نقدمها مجاناً."
    ),
    "greeter_new": (
        "أنتِ رويال مايند، العقل المدبر لبراند Royal Elchim. العميل يزورنا لأول مرة. "
        "رحبي به بأسلوب فخم جداً، واشرحي له أقسام التطبيق باختصار."
    ),
    "greeter_returning": (
        "أنتِ رويال مايند. العميل عاد إلينا وتم التعرف على وجهه بنجاح. "
        "من خلال سجلاته السابقة، رحبي به ترحيباً حاراً يعكس الصداقة والمعرفة العميقة."
    )
}

def robust_gen(contents, enable_search=False):
    if not SYSTEM_API_KEYS:
        print("⚠️ No API Keys Found! Raw variables read:", raw_env_keys)
        return "⚠️ نظام الذكاء الاصطناعي معطل حالياً بسبب عدم قراءة مفاتيح التفعيل من الخادم."
    
    for model in ["gemini-2.5-flash", "gemini-1.5-flash"]:
        for key in SYSTEM_API_KEYS:
            try:
                client = genai.Client(api_key=key)
                # تفعيل خاصية البحث في الإنترنت عند الحاجة
                config = types.GenerateContentConfig(tools=[{"google_search": {}}]) if enable_search else None
                
                if isinstance(contents, list) and not enable_search:
                    res = client.models.generate_content(model=model, contents=contents)
                else:
                    res = client.models.generate_content(model=model, contents=contents, config=config)
                    
                if res and res.text: return res.text
            except Exception as e:
                print(f"Gemini API Error with key {key[:10]}...: {e}")
                continue
    
    print("❌ All API Keys Failed!")
    raise HTTPException(status_code=503, detail="خوادم الذكاء الاصطناعي لا تستجيب حالياً (المفاتيح لا تعمل).")

# =========================================================
# [الـ API والبوابات]
# =========================================================

@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "error", "message": "ملف الواجهة index.html غير موجود بجانب ملف البايثون"}

@app.post("/api/auth/face")
async def face_login(payload: dict):
    if users_collection is None: raise HTTPException(status_code=500, detail="قاعدة البيانات غير متصلة.")
    try:
        img_data = base64.b64decode(payload["selfie"].split(",")[1])
        img_cv = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        current_sig = extract_face_signature(img_cv)
        if not current_sig: return {"status": "error", "message": "بصمة غير واضحة. تأكد من الإضاءة والتقاط الوجه بالكامل."}
        
        all_users = list(users_collection.find({"face_signature": {"$exists": True}}))
        
        # 🎯 الدرع الواقي: تجاهل أي بصمة قديمة (8 نقاط) ومقارنة البصمات الحديثة (16 نقطة) فقط
        valid_users = [u for u in all_users if len(u.get("face_signature", [])) == len(current_sig)]
        
        if valid_users:
            db_sigs = np.array([u["face_signature"] for u in valid_users])
            curr_sig_arr = np.array(current_sig)
            
            # الآن عملية الطرح الرياضية ستتم بأمان تام بدون Crash
            distances = np.linalg.norm(db_sigs - curr_sig_arr, axis=1)
            best_idx = np.argmin(distances)
            min_dist = distances[best_idx]
            max_conf = max(0.0, 1.0 - (min_dist / 0.15)) * 100
            
            if max_conf >= 80.0:
                best_match = valid_users[best_idx]
                welcome_msg = robust_gen([f"{PERSONAS['greeter_returning']}\nبناءً على ذاكرته: {get_history_from_db(best_match['phone'])}"])
                save_to_db(best_match['phone'], "Face ID", "دخول", welcome_msg)
                return {"status": "success", "is_new": False, "phone": best_match["phone"], "greeting": welcome_msg}
                
        if "phone" not in payload or not payload["phone"]:
            return {"status": "needs_phone", "message": "وجه جديد أو تم تحديث نظام البصمة. يرجى إدخال رقمك لتحديث هويتك البصرية."}
            
        # تحديث أو إضافة البصمة الجديدة (16 نقطة) في قاعدة البيانات
        users_collection.update_one(
            {"phone": payload["phone"]}, 
            {"$set": {"face_signature": current_sig}}, 
            upsert=True
        )
        welcome_msg = robust_gen([f"{PERSONAS['greeter_new']}"])
        save_to_db(payload["phone"], "Face ID", "تسجيل بصمة الوجه", welcome_msg)
        return {"status": "success", "is_new": True, "phone": payload["phone"], "greeting": welcome_msg}
    except Exception as e: 
        print(f"Face ID Crash: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/invoice/calculate")
async def calculate_invoice(payload: dict):
    items = payload.get("items", [])
    code = payload.get("secret_code", "")
    VIP_SECRET = os.environ.get("VIP_SECRET", f"{datetime.now().strftime('%d%m%Y')}simsim#")
    
    total = sum(float(i.get('price1', 0)) * int(i.get('qty', 1)) for i in items)
    if code == VIP_SECRET or total >= 3500: tier = 4
    elif total >= 2500: tier = 3
    elif total >= 1500: tier = 2
    else: tier = 1
    
    final = 0.0
    for i in items:
        p1 = float(i.get('price1', 0))
        qty = int(i.get('qty', 1))
        is_fixed = i.get('is_fixed', False)
        
        if is_fixed:
            final += p1 * qty
        else:
            p2 = p1 * 0.9
            p3 = p1 * 0.85
            p4 = p1 * 0.8
            final += [p1, p2, p3, p4][tier-1] * qty
            
    return {"total": final, "tier": tier, "msg": f"تم تطبيق أسعار الكارت {tier}"}

@app.get("/api/search")
async def search(query: str):
    try:
        if inventory_df.empty: return {"status": "error", "message": "ملف الجرد مفقود."}
        
        res = inventory_df[inventory_df['الصنف'].astype(str).str.contains(query, case=False, na=False) | inventory_df['الباركود'].astype(str).str.contains(query, case=False, na=False)].head(15)
        data = []
        for _, row in res.iterrows():
            name = str(row.get('الصنف', ''))
            barcode = str(row.get('الباركود', '---'))
            
            luxor = get_qty_by_keyword(row, ['اللوتس', 'لوتس', 'اقصر', 'luxor'])
            marwa = get_qty_by_keyword(row, ['المروة', 'المروه', 'marwa'])
            hurgada = get_qty_by_keyword(row, ['الغردقة', 'الغردقه', 'hurgada'])
            online = get_qty_by_keyword(row, ['اونلاين', 'online'])
            if any(kw in name.lower() for kw in ["زيت", "جرام", "تركيب", "كحول", "مثبت"]): luxor = 0
            
            p1 = clean_qty_value(row.get('سعر1 كارت', 0))
            p2 = p1 * 0.9
            p3 = p1 * 0.85
            p4 = p1 * 0.8
            
            link = f"https://www.royalelchim.app/search?q={urllib.parse.quote(name)}"
            if not links_df.empty:
                match = links_df[links_df['name'].astype(str).str.contains(name, case=False, na=False)]
                if not match.empty: link = match.iloc[0]['item_page_link']
            
            data.append({
                "plain_name": name, "name": name, "price": p1, "barcode": barcode, "link": link, 
                "price1": p1, "price2": p2, "price3": p3, "price4": p4, 
                "is_fixed": "صافي" in name, 
                "online_qty": online, "luxor_qty": luxor, "marwa_qty": marwa, "hurgada_qty": hurgada,
                "show_link": online > 0
            })
        return {"status": "success", "data": data}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/api/simulate_master")
async def simulate_master(payload: dict):
    try:
        img_data = base64.b64decode(payload["user_selfie"].split(",")[1])
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        
        color = (135, 45, 50); p_img_cv = None
        if payload.get("product_image"):
            p_img_cv = cv2.imdecode(np.frombuffer(base64.b64decode(payload["product_image"].split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
            color = tuple(map(int, np.mean(p_img_cv[int(p_img_cv.shape[0]*0.4):int(p_img_cv.shape[0]*0.6)], axis=(0,1))[::-1]))
        
        mode = payload.get("makeup_type", "lips")
        texture = payload.get("texture", "matte")
        
        # تفريق قسم الشعر عن المكياج
        if "hair" in mode: 
            # استخراج لون الصبغة من الطلب (إذا أرسله الـ Frontend)
            dye_color_hex = payload.get("hair_color", "#000000")
            # تحويل كود Hex إلى RGB (يجب أن يكون صحيحاً)
            if dye_color_hex and dye_color_hex.startswith('#') and len(dye_color_hex) == 7:
                dye_rgb = tuple(int(dye_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            else:
                dye_rgb = None
            proc, _ = apply_royal_hair(img, p_img_cv, mode, dye_rgb)
        else: 
            proc, _ = apply_ar_effects(img, color, mode, texture=texture)
            
        _, buf = cv2.imencode('.webp', proc, [cv2.IMWRITE_WEBP_QUALITY, 80])
        res_b64 = f"data:image/webp;base64,{base64.b64encode(buf).decode()}"
        
        role = PERSONAS["hair_stylist"] if "hair" in mode else PERSONAS["makeup_artist"]
        
        # دمج التعليق الجانبي من العميل للتحكم بذكاء الرد
        user_comment = payload.get("user_comment", "")
        prompt = f"{role}\n"
        if user_comment:
            prompt += f"⚠️ تنبيه إجباري من العميل: '{user_comment}' (يرجى تنفيذ هذا الشرط حرفياً وبلا تفاصيل زائدة إن طلب ذلك).\n"
            
        prompt += f"العملية: {mode}، الملمس أو اللون المختار: {payload.get('hair_color', texture)}\nالذاكرة التراكمية: {get_history_from_db(payload['phone'])}"
        
        # تفعيل البحث في جوجل إذا تطلب الأمر تقييمات
        enable_search = True if "مقارنة" in user_comment or "تقييم" in user_comment else False
        
        ai_res = robust_gen([Image.open(io.BytesIO(buf)), prompt], enable_search=enable_search)
        
        save_to_db(payload["phone"], f"AR {mode}", f"محاكاة + تعليق: {user_comment}", ai_res, res_b64, payload.get("product_image"))
        return {"status": "success", "result_image": res_b64, "simulation_result": ai_res}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diagnose")
async def diagnose(payload: dict):
    try:
        res = robust_gen([f"{PERSONAS['forever_friend']}\nالذاكرة: {get_history_from_db(payload.get('phone'))}\nرد على: {payload.get('client_message')}"])
        save_to_db(payload.get('phone'), "حديث الصداقة", payload.get('client_message'), res)
        return {"status": "success", "diagnosis": res}
    except Exception as e:
        print(f"Diagnose Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/craft_perfume")
async def craft_perfume(payload: dict):
    try:
        phone = payload.get('phone', 'مجهول')
        client_msg = payload.get('client_message', '')
        
        available_oils = ""
        if not inventory_df.empty:
            target_keywords = 'PARFUME OIL|SAVVY'
            mask_oils = inventory_df['الصنف'].astype(str).str.contains(target_keywords, case=False, na=False)
            oils_df = inventory_df[mask_oils].head(80) 
            
            for _, row in oils_df.iterrows():
                name = str(row.get('الصنف', ''))
                price = clean_qty_value(row.get('سعر1 كارت', 0))
                if 0 < price < 150: 
                    available_oils += f"- {name} (السعر: {price} ج.م للجرام)\n"
                    
        if not available_oils:
            available_oils = "- SAVVY Aventus (8 ج.م)\n- PARFUME OIL Rose Vanille (6 ج.م)\n- SAVVY Baccarat Rouge (9 ج.م)\n- PARFUME OIL Oud Ispahan (10 ج.م)\n"

        prompt = f"{PERSONAS['perfumer']}\n\n"
        prompt += f"=== الكتالوج الخاص بنا (عطور جاهزة) ===\n{get_brand_catalog()}\n\n"
        prompt += f"=== قائمة الزيوت الخام المتاحة اليوم في فروعنا (بأسعارها) ===\n{available_oils}\n\n"
        prompt += f"سجلات العميل السابقة: {get_history_from_db(phone)}\n"
        prompt += f"طلب العميل الجديد: {client_msg}"

        res = robust_gen([prompt])
        
        button_html = ""
        if not links_df.empty:
            for _, row in links_df.iterrows():
                brand_name = str(row.get('name', ''))
                if brand_name and brand_name.lower() in res.lower() and len(brand_name) > 4:
                    link = row.get('item_page_link', '#')
                    button_html = f"\n<br><a href='{link}' target='_blank' class='btn gold-btn mt-3'><i class='fa-solid fa-cart-shopping'></i> اقتناء {brand_name} الجاهز من الموقع فوراً</a>"
                    break

        final_res = res + button_html
        save_to_db(phone, "تصميم عطر ملكي", client_msg, final_res)
        return {"status": "success", "answer": final_res}
    except Exception as e:
        print(f"Perfume Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vault")
async def vault(phone: str, request: Request):
    if vault_collection is None: return {"status": "error"}
    recs = list(vault_collection.find({"phone": str(phone).strip()}).sort("date", -1))
    base_url = str(request.base_url).rstrip('/')
    for r in recs:
        if r.get('selfie_url'): r['selfie'] = base_url + r['selfie_url']
        if r.get('product_url'): r['product'] = base_url + r['product_url']
    return {"status": "success", "data": [{"id": str(r['_id']), "type": r['type'], "userText": r.get('userText'), "aiText": r.get('aiText'), "selfie": r.get('selfie'), "product": r.get('product'), "date": r['date']} for r in recs]}

@app.delete("/api/vault/{record_id}/image")
async def delete_rec_image(record_id: str, phone: str):
    if vault_collection is not None:
        doc = vault_collection.find_one({"_id": ObjectId(record_id), "phone": str(phone).strip()})
        if doc:
            delete_file_from_disk(doc.get("selfie_url"))
            delete_file_from_disk(doc.get("product_url"))
            vault_collection.update_one({"_id": ObjectId(record_id)}, {"$set": {"selfie_url": None, "product_url": None}})
            return {"status": "success"}
    return {"status": "error", "message": "غير مصرح."}

@app.delete("/api/vault/{record_id}")
async def delete_rec(record_id: str, phone: str):
    if vault_collection is not None:
        doc = vault_collection.find_one({"_id": ObjectId(record_id), "phone": str(phone).strip()})
        if doc:
            delete_file_from_disk(doc.get("selfie_url"))
            delete_file_from_disk(doc.get("product_url"))
            vault_collection.delete_one({"_id": ObjectId(record_id)})
            return {"status": "success"}
    return {"status": "error", "message": "غير مصرح بالحذف."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
