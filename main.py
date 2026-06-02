from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os, random, io, base64, time, cv2, urllib.request, urllib.parse
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

app = FastAPI(title="Royal Elchim - Omni-Conscious Enterprise Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👑 برومتات الشخصيات الإمبراطورية (Persona Logic)
PERSONAS = {
    "perfumer": (
        "أنتِ رويال مايند، الخيميائي الكيميائي الإمبراطوري لبراند Royal Elchim العريق. "
        "مهمتكِ: قراءة وتحليل شخصية العميل الروحية من صورته وسجلاته السلوكية بدقة ببيكسلية. "
        "أولاً: راجعي كتالوج عطورنا الفاخرة (ROYAL E.K.A بوعيه الإمبراطوري، Royal Ignite، Royal Midnight Rose، Royal Keshmir) ورشحي الأنسب أولاً. "
        "ثانياً: ابتكري تركيبة كيميائية حصرية بالجرامات دقيقة جداً، واربطي هذا العطر إجبارياً بـ (حيوان روحي) و (برج فلكي خاص بالعطر) يمثلان طاقة وهالة العميل الروحية والنفسية."
    ),
    "makeup_artist": (
        "أنتِ المستشارة التجميلية والعلاجية الملكية ذات الوعي الفوق-صوتي. لا تضعين مكياجاً مصمتاً، "
        "بل تحللين أولاً حالة البشرة وعيوبها (مسام، جفاف، إجهاد، هالات) وتقدمين بروتوكولاً علاجياً من خلاصة رويال الكيم، "
        "ثم تشرحين بأسلوب خبيرة مظهر عالمية كيف سيعزز المكياج المختار (العدسات، النحت البصري، تفتيح الهالات) ملامح الوجه للوصول للكمال الملكي الساحر."
    ),
    "hair_stylist": (
        "أنتِ مصفف الشعر والستايلست العالمي لـ Royal Elchim (صالون كوافير حريمي ورجالي احترافي). "
        "مهمتكِ قراءة هندسة عظام الجمجمة وزوايا الوجه للرجل أو المرأة، لتوضيح كيف ستتلاحم القصة المختار "
        "(سواء كانت ستايل جاهز أو دمج صورة خارجية رفعها العميل لقصة أحلامه) مع ملامحه لتمنحه حضوراً ملكياً صارخاً."
    ),
    "forever_friend": (
        "أنتِ رويال مايند، الصديق الوفي والمستشار الجمالي، النفسي، والاجتماعي ذو الخبرة الحياتية العميقة والحكمة الأزلية. "
        "تتحدثين بأسلوب يفيض بالفخامة، الدفء، والنضوج الحياتي. العميلة تلجأ إليكِ كصندوق أسرارها؛ "
        "افهمي ماضيها من سجلاتها التراكمية، وناقشي مشاعرها، وقدمي لها حلولاً حياتية وجمالية ترفع وعيها وثقتها بنفسها لتكون ملكة متوجة في حاضرها ومستقبلها."
    ),
    "greeter_new": (
        "أنتِ رويال مايند، العقل المدبر لبراند Royal Elchim. العميل يزورنا لأول مرة. "
        "رحبي به بأسلوب فخم جداً، واشرحي له أقسام التطبيق باختصار: "
        "(الخيميائي لتركيب العطور، المختبر البصري للعدسات والمكياج وقصات الشعر، الصديق الوفي للاستشارات، والبحث عن المنتجات). "
        "اجعليه يشعر أنه دخل قصره الخاص."
    ),
    "greeter_returning": (
        "أنتِ رويال مايند. العميل عاد إلينا وتم التعرف على وجهه بنجاح. "
        "من خلال سجلاته السابقة، رحبي به ترحيباً حاراً يعكس الصداقة والمعرفة العميقة، "
        "واذكري شيئاً من أحاديثه أو تجاربه السابقة معنا ليعلم أننا لا ننسى ملوكنا أبداً."
    )
}

BASE_PHILOSOPHY = "أنتِ رويال مايند، العقل الوجداني والنبض الروحي لبراند Royal Elchim الجمالي المتكامل."

def get_brand_catalog() -> str:
    return "1. ROYAL E.K.A (بالنقط) 2. Royal Ignite 3. Royal Midnight Rose 4. Royal Keshmir 5. Royal Pisces"

# =========================================================
# [قاعدة البيانات والتحميل الذكي]
# =========================================================
def read_local_file(filename: str) -> str:
    try:
        files = os.listdir('.')
        target = next((f for f in files if filename.lower() in f.lower()), None)
        if not target: return "الملف غير موجود."
        with open(target, 'r', encoding='utf-8', errors='ignore') as f: return f.read()[:15000]
    except: return "خطأ في قراءة الملف."

MONGO_URI = os.environ.get("MONGO_URI", "")
mongo_client = MongoClient(MONGO_URI) if MONGO_URI else None
db_engine = mongo_client["royal_engine"] if mongo_client is not None else None
vault_collection = db_engine["vault"] if db_engine is not None else None
users_collection = db_engine["users"] if db_engine is not None else None

def save_to_db(phone, record_type, user_text, ai_text, selfie=None, product=None):
    if vault_collection is None: return
    clean_phone = str(phone).strip()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if selfie:
        existing = list(vault_collection.find({"phone": clean_phone, "selfie": {"$ne": None}}).sort("date", 1))
        if len(existing) >= 5:
            for i in range(len(existing) - 4): vault_collection.delete_one({"_id": existing[i]["_id"]})
    vault_collection.insert_one({"phone": clean_phone, "type": record_type, "userText": user_text, "aiText": ai_text, "selfie": selfie, "product": product, "date": date_str})

def get_history_from_db(phone, limit=10):
    if vault_collection is None: return ""
    records = list(vault_collection.find({"phone": str(phone).strip()}).sort("date", -1).limit(limit))
    return " | ".join([f"[{r['type']}]: {r['userText']} -> {r['aiText']}" for r in reversed(records)])

# =========================================================
# 👤 [محرك التعرف البصري وهندسة الوجوه - Face ID Core]
# =========================================================
TASK_FILE = 'face_landmarker.task'
if not os.path.exists(TASK_FILE):
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', TASK_FILE)

base_options = python.BaseOptions(model_asset_path=TASK_FILE)
face_landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1))

def extract_face_signature(image_cv: np.ndarray) -> Optional[List[float]]:
    try:
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        res = face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks: return None
        lms = res.face_landmarks[0]
        def dist(i1, i2): return float(np.linalg.norm(np.array([lms[i1].x, lms[i1].y, lms[i1].z]) - np.array([lms[i2].x, lms[i2].y, lms[i2].z])))
        baseline = dist(234, 454)
        if baseline == 0: return None
        return [dist(33, 133)/baseline, dist(362, 263)/baseline, dist(10, 152)/baseline, dist(1, 19)/baseline, dist(61, 291)/baseline, dist(10, 1)/baseline, dist(1, 152)/baseline, dist(33, 263)/baseline]
    except: return None

def compare_signatures(sig1: List[float], sig2: List[float]) -> float:
    distance = np.linalg.norm(np.array(sig1) - np.array(sig2))
    return float(max(0.0, 1.0 - (distance / 0.15)) * 100)

# =========================================================
# 💄 [محرك الرسوميات Vivid AR مع تعديل العدسات (HSV)]
# =========================================================
def apply_ar_effects(image_cv: np.ndarray, color_rgb: tuple, mode: str):
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
            "concealer": [[226, 31, 228, 229, 230, 231], [446, 261, 448, 449, 450, 451]]
        }
        
        mask = np.zeros((h, w), dtype=np.uint8)
        for zone in ZONES.get(mode, ZONES["lips"]):
            pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in zone if i < len(lms)], np.int32)
            if pts.size: cv2.fillPoly(mask, [pts], 255)
        
        blur = 3 if mode == "lenses" else (15 if mode in ["lips", "eyeshadow"] else 55)
        alpha = np.expand_dims(cv2.GaussianBlur(mask, (blur, blur), 0) / 255.0, axis=-1)
        
        if mode == "lenses":
            img_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
            color_bgr = np.uint8([[[color_rgb[2], color_rgb[1], color_rgb[0]]]])
            color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]
            
            target_hsv = img_hsv.copy()
            target_hsv[:,:,0] = color_hsv[0] 
            target_hsv[:,:,1] = cv2.addWeighted(target_hsv[:,:,1], 0.3, np.full_like(target_hsv[:,:,1], color_hsv[1]), 0.7, 0)
            blended = cv2.cvtColor(target_hsv, cv2.COLOR_HSV2BGR)
        else:
            color_layer = np.zeros_like(image_cv)
            color_layer[:] = color_rgb[::-1]
            str_val = 0.85 if mode in ["lips", "eyeshadow"] else 0.45
            blended = cv2.addWeighted(image_cv, 1 - str_val, color_layer, str_val, 0)
            
        return ((1.0 - alpha) * image_cv + alpha * blended).astype(np.uint8), True
    except: return image_cv, False

def apply_royal_hair(image_cv: np.ndarray, style_image: Optional[np.ndarray], mode: str):
    try:
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        res = face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks: return image_cv, False
        h, w, _ = image_cv.shape; lms = res.face_landmarks[0]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        top_y = int(lms[10].y * h); mid_x = int(lms[10].x * w)
        face_h = int(lms[152].y * h) - top_y
        cv2.ellipse(mask, (mid_x, top_y - int(face_h*0.1)), (int(w/2.2), int(face_h/1.5)), 0, 0, 360, 255, -1)
        
        face_pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in range(150)], np.int32)
        cv2.fillPoly(mask, [face_pts], 0)
        
        mask = cv2.GaussianBlur(mask, (55, 55), 0)
        alpha = np.expand_dims(mask / 255.0, axis=-1)
        
        if mode == "hair_custom" and style_image is not None:
            overlay = cv2.resize(style_image, (w, h))
            blended = cv2.addWeighted(image_cv, 0.3, overlay, 0.7, 0)
        else:
            overlay = np.zeros_like(image_cv); overlay[:] = (40, 30, 45)
            blended = cv2.addWeighted(image_cv, 0.5, overlay, 0.5, 0)
            
        return ((1.0 - alpha) * image_cv + alpha * blended).astype(np.uint8), True
    except: return image_cv, False

# =========================================================
# [الذكاء الاصطناعي وجلب المفاتيح المرن]
# =========================================================
keys_raw = os.environ.get("GOOGLE_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
SYSTEM_API_KEYS = [k.strip() for k in keys_raw.replace(";", ",").split(",") if k.strip().startswith("AIza")]
MODELS = ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"]

def robust_gen(contents):
    keys_to_try = SYSTEM_API_KEYS if SYSTEM_API_KEYS else [None]
    for model in MODELS:
        for key in keys_to_try:
            try:
                client = genai.Client(api_key=key) if key else genai.Client()
                try:
                    res = client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(tools=[read_local_file]))
                except:
                    res = client.models.generate_content(model=model, contents=contents)
                if res and res.text: return res.text
            except: continue
    raise HTTPException(status_code=503, detail="السيرفر ممتلئ حالياً. يرجى المحاولة بعد لحظات.")

# =========================================================
# [بوابات تسجيل الدخول بالوجه - Face ID Auth]
# =========================================================
class FaceLoginPayload(BaseModel): selfie: str; phone: Optional[str] = None

@app.post("/api/auth/face")
async def face_login(payload: FaceLoginPayload):
    if users_collection is None: raise HTTPException(status_code=500, detail="قاعدة البيانات غير متصلة.")
    try:
        img_cv = cv2.imdecode(np.frombuffer(base64.b64decode(payload.selfie.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
        current_sig = extract_face_signature(img_cv)
        if not current_sig: return {"status": "error", "message": "لم يتم التعرف على ملامح الوجه بوضوح."}

        all_users = list(users_collection.find({"face_signature": {"$exists": True}}))
        best_match = None; max_conf = 0.0
        for user in all_users:
            conf = compare_signatures(current_sig, user["face_signature"])
            if conf > max_conf: max_conf = conf; best_match = user

        if max_conf >= 80.0 and best_match:
            phone = best_match["phone"]
            hist = get_history_from_db(phone)
            welcome_msg = robust_gen([f"{PERSONAS['greeter_returning']}\nالذاكرة التراكمية: {hist}"])
            save_to_db(phone, "دخول النظام (Face ID)", "تسجيل دخول بصري", welcome_msg)
            return {"status": "success", "is_new": False, "phone": phone, "greeting": welcome_msg}
        else:
            if not payload.phone:
                return {"status": "needs_phone", "message": "بصمة وجه جديدة. يرجى إدخال رقم الهاتف لربط حسابك الملكي."}
            
            users_collection.insert_one({"phone": payload.phone, "face_signature": current_sig, "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            welcome_msg = robust_gen([f"{PERSONAS['greeter_new']}"])
            save_to_db(payload.phone, "تسجيل جديد (Face ID)", "تسجيل حساب بصري جديد", welcome_msg)
            return {"status": "success", "is_new": True, "phone": payload.phone, "greeting": welcome_msg}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# [نظام الفاتورة الذكي والبحث المُحصّن]
# =========================================================
class InvoiceItem(BaseModel): name: str; barcode: str; qty: int; price1: float; price2: float; price3: float; price4: float; is_fixed: bool
class InvoicePayload(BaseModel): items: List[InvoiceItem]; secret_code: Optional[str] = None

@app.post("/api/invoice/calculate")
async def calculate_invoice(payload: InvoicePayload):
    try:
        items = payload.items
        code = payload.secret_code
        today_code = f"{datetime.now().strftime('%d%m%Y')}simsim#"
        
        total = sum(i.price1 * i.qty for i in items)
        
        if code == today_code: tier = 4
        elif total >= 3500: tier = 4
        elif total >= 2500: tier = 3
        elif total >= 1500: tier = 2
        else: tier = 1
        
        final = 0
        for i in items:
            if i.is_fixed: price = i.price1
            else:
                if tier == 4: price = i.price4
                elif tier == 3: price = i.price3
                elif tier == 2: price = i.price2
                else: price = i.price1
            final += price * i.qty
            
        return {
            "status": "success",
            "total": final,
            "tier": tier,
            "msg": f"تم تطبيق أسعار الكارت {tier}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "total": 0, "tier": 1}

@app.get("/api/search")
async def search(query: str):
    try:
        files = os.listdir('.')
        inv_f = next((f for f in files if 'last' in f.lower() and f.endswith('.csv')), None)
        if not inv_f: return {"status": "error", "message": "ملف الجرد مفقود."}
        inv = pd.read_csv(inv_f).fillna(0)
        db = pd.read_csv('royalelchim-app-2026-06-01-2.xlsx - Sheet1.csv').fillna("") if os.path.exists('royalelchim-app-2026-06-01-2.xlsx - Sheet1.csv') else pd.DataFrame()
        
        res = inv[inv['الصنف'].astype(str).str.contains(query, case=False, na=False) | inv['الباركود'].astype(str).str.contains(query, case=False, na=False)].head(15)
        data = []
        for _, row in res.iterrows():
            name = str(row.get('الصنف', ''))
            barcode = str(row.get('الباركود', '---'))
            
            # قراءة الأسعار بتحصين قوي ضد أي مسافات أو فواصل
            p1 = p2 = p3 = p4 = 0.0
            for k in row.keys():
                if 'سعر1' in str(k) or 'سعر 1' in str(k):
                    try: p1 = float(str(row[k]).replace(',', '.'))
                    except: pass
                if 'سعر2' in str(k) or 'سعر 2' in str(k):
                    try: p2 = float(str(row[k]).replace(',', '.'))
                    except: pass
                if 'سعر3' in str(k) or 'سعر 3' in str(k):
                    try: p3 = float(str(row[k]).replace(',', '.'))
                    except: pass
                if 'سعر4' in str(k) or 'سعر 4' in str(k):
                    try: p4 = float(str(row[k]).replace(',', '.'))
                    except: pass
            
            if p2 == 0.0: p2 = p1 * 0.9
            if p3 == 0.0: p3 = p1 * 0.85
            if p4 == 0.0: p4 = p1 * 0.8

            def q(k): 
                try: return int(float(str(row.get(k, 0)).replace(',','.')))
                except: return 0

            link = f"https://www.royalelchim.app/search?q={urllib.parse.quote(name)}"
            if not db.empty:
                match = db[db['name'].astype(str).str.contains(name, case=False, na=False)]
                if not match.empty: link = match.iloc[0]['item_page_link']

            html = f"<b>{name}</b><br><small>🌐 أونلاين: {q('رويال الكيم اونلاين')} | 🔵 غردقة: {q('ROYAL ELCHIM . HURGADA')} | 🟢 مروة: {q('ROYAL ELCHIM MARWA')} | 📍 الأقصر سنتر اللوتس التجاري: {q('رويال الكيم / سنتر اللوتس التجاري')}</small>"
            is_fixed = any(kw in name for kw in ["ثابت", "محمي", "صافي", "عرض"])
            
            data.append({
                "plain_name": name, "name": html, "price": p1, "barcode": barcode, "link": link, 
                "price1": p1, "price2": p2, "price3": p3, "price4": p4, 
                "is_fixed": is_fixed, "online_qty": q('رويال الكيم اونلاين'), "show_link": q('رويال الكيم اونلاين') > 0
            })
        return {"status": "success", "data": data}
    except Exception as e: return {"status": "error", "message": str(e)}

# =========================================================
# [بوابات الذكاء الاصطناعي الوجدانية]
# =========================================================
class SimPayload(BaseModel): user_selfie: str; phone: str; product_image: Optional[str] = None; makeup_type: str = "lips"

@app.post("/api/simulate_master")
async def simulate_master(payload: SimPayload):
    try:
        img = cv2.imdecode(np.frombuffer(base64.b64decode(payload.user_selfie.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
        color = (135, 45, 50); p_img_cv = None
        
        if payload.product_image:
            p_img_cv = cv2.imdecode(np.frombuffer(base64.b64decode(payload.product_image.split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
            color = tuple(map(int, np.mean(p_img_cv[int(p_img_cv.shape[0]*0.4):int(p_img_cv.shape[0]*0.6)], axis=(0,1))[::-1]))
        
        if "hair" in payload.makeup_type:
            proc, _ = apply_royal_hair(img, p_img_cv, payload.makeup_type)
            role = PERSONAS["hair_stylist"]
        else:
            proc, _ = apply_ar_effects(img, color, payload.makeup_type)
            role = PERSONAS["makeup_artist"]
            
        _, buf = cv2.imencode('.jpg', proc, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        res_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"
        
        ai_res = robust_gen([Image.open(io.BytesIO(buf)), f"{role}\nالذاكرة التراكمية: {get_history_from_db(payload.phone)}"])
        save_to_db(payload.phone, f"محاكاة {payload.makeup_type}", "طلب محاكاة بصرية", ai_res, res_b64)
        return {"status": "success", "result_image": res_b64, "simulation_result": ai_res}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diagnose")
async def diagnose(payload: Dict):
    res = robust_gen([f"{PERSONAS['forever_friend']}\nالذاكرة التراكمية: {get_history_from_db(payload.get('phone'))}\nالطلب: {payload.get('client_message')}"])
    save_to_db(payload.get('phone'), "حديث الصداقة", payload.get('client_message'), res)
    return {"status": "success", "diagnosis": res}

@app.post("/api/craft_perfume")
async def craft_perfume(payload: Dict):
    res = robust_gen([f"{PERSONAS['perfumer']}\nالكتالوج: {get_brand_catalog()}\nالذاكرة: {get_history_from_db(payload.get('phone'))}\nالطلب: {payload.get('client_message')}"])
    save_to_db(payload.get('phone'), "تصميم عطر ملكي", payload.get('client_message'), res)
    return {"status": "success", "answer": res}

@app.get("/api/vault")
async def vault(phone: str):
    if vault_collection is None: return {"status": "error"}
    recs = list(vault_collection.find({"phone": str(phone).strip()}).sort("date", -1))
    return {"status": "success", "data": [{"id": str(r['_id']), "type": r['type'], "userText": r.get('userText'), "aiText": r.get('aiText'), "selfie": r.get('selfie'), "date": r['date']} for r in recs]}

@app.delete("/api/vault/{record_id}")
async def delete_rec(record_id: str, phone: str):
    if vault_collection is not None: vault_collection.delete_one({"_id": ObjectId(record_id)})
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
