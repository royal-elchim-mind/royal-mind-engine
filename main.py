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

# --- مكتبات المتصفح الآلي (Selenium) ---
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# =========================================================
# [1] إعدادات النظام والأمان (Enterprise Setup)
# =========================================================

app = FastAPI(title="Royal Elchim - Enterprise V5")

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
    if vault_collection is None: return
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
    except Exception as e: print(f"DB Save Error: {e}")

def get_history_from_db(phone, limit=10):
    if vault_collection is None: return "لا توجد سجلات."
    records = list(vault_collection.find({"phone": str(phone).strip()}).sort("date", -1).limit(limit))
    return " | ".join([f"[{r['type']}]: العميل: {r.get('userText','')} -> رويال: {r.get('aiText','')}" for r in reversed(records)])

# =========================================================
# [3] محركات التعرف البصري والفصل (Vision AI Models)
# =========================================================

TASK_FILE = 'face_landmarker.task'
if not os.path.exists(TASK_FILE):
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', TASK_FILE)

face_landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=TASK_FILE), num_faces=1))

SEG_TASK_FILE = 'selfie_multiclass_256x256.tflite'
if not os.path.exists(SEG_TASK_FILE):
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite', SEG_TASK_FILE)

seg_options = vision.ImageSegmenterOptions(base_options=python.BaseOptions(model_asset_path=SEG_TASK_FILE), output_category_mask=True)
image_segmenter = vision.ImageSegmenter.create_from_options(seg_options)

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
# [4] محرك الرسوميات Vivid AR ومحرك الشعر الذكي
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
            color_layer = np.zeros_like(image_cv); color_layer[:] = color_rgb[::-1]
            blended = cv2.addWeighted(blended, 0.85, color_layer, 0.15, 0)
        else:
            for zone in ZONES.get(mode, ZONES["lips"]):
                pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in zone if i < len(lms)], np.int32)
                if pts.size: cv2.fillPoly(mask, [pts], 255)
            blur = 15 if mode in ["lips", "eyeshadow", "tint"] else 55
            mask = cv2.GaussianBlur(mask, (blur, blur), 0)
            alpha = np.expand_dims(mask / 255.0, axis=-1)
            color_layer = np.zeros_like(image_cv); color_layer[:] = color_rgb[::-1]
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
        
        p127 = np.array([lms[127].x * w, lms[127].y * h]); p356 = np.array([lms[356].x * w, lms[356].y * h])
        face_width = max(10, np.linalg.norm(p127 - p356)) 
        p10 = np.array([lms[10].x * w, lms[10].y * h])
        anchor_x = int(p10[0]); anchor_y = int(p10[1])
        
        if style_image is None: return image_cv, False
        style_rgb = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
        mp_style_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=style_rgb)
        
        segmentation_result = image_segmenter.segment(mp_style_image)
        category_mask = segmentation_result.category_mask.numpy_view()
        hair_mask_binary = (category_mask == 1).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hair_mask_clean = cv2.morphologyEx(hair_mask_binary, cv2.MORPH_CLOSE, kernel)
        hair_mask_clean = cv2.GaussianBlur(hair_mask_clean, (5, 5), 0)
        
        ys, xs = np.where(hair_mask_clean > 50)
        if len(ys) == 0 or len(xs) == 0: return image_cv, False
        y1, y2 = np.min(ys), np.max(ys); x1, x2 = np.min(xs), np.max(xs)
        
        cropped_hair_rgb = style_image[y1:y2, x1:x2].copy()
        cropped_hair_mask = hair_mask_clean[y1:y2, x1:x2]
        
        if color_rgb is not None:
            style_hsv = cv2.cvtColor(cropped_hair_rgb, cv2.COLOR_BGR2HSV)
            target_bgr = np.uint8([[[color_rgb[2], color_rgb[1], color_rgb[0]]]])
            target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
            dyed_style_hsv = style_hsv.copy()
            dyed_style_hsv[:, :, 0] = target_hsv[0]
            dyed_style_hsv[:, :, 1] = cv2.addWeighted(dyed_style_hsv[:, :, 1], 0.3, np.full_like(dyed_style_hsv[:, :, 1], target_hsv[1]), 0.7, 0)
            cropped_hair_rgb = cv2.cvtColor(dyed_style_hsv, cv2.COLOR_HSV2BGR)

        ch, cw, _ = cropped_hair_rgb.shape
        target_hair_w = max(10, int(face_width * 1.5))
        scale_ratio = target_hair_w / max(1, cw)
        target_hair_h = max(10, int(ch * scale_ratio))
        
        resized_hair = cv2.resize(cropped_hair_rgb, (target_hair_w, target_hair_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(cropped_hair_mask, (target_hair_w, target_hair_h), interpolation=cv2.INTER_AREA)
        
        start_x = anchor_x - (target_hair_w // 2)
        start_y = anchor_y - int(target_hair_h * 0.85)
        
        blended = image_cv.copy()
        bg_y1 = max(0, start_y); bg_y2 = min(h, start_y + target_hair_h)
        bg_x1 = max(0, start_x); bg_x2 = min(w, start_x + target_hair_w)
        fg_y1 = max(0, -start_y); fg_y2 = fg_y1 + (bg_y2 - bg_y1)
        fg_x1 = max(0, -start_x); fg_x2 = fg_x1 + (bg_x2 - bg_x1)
        
        if (bg_y2 > bg_y1) and (bg_x2 > bg_x1) and (fg_y2 <= target_hair_h) and (fg_x2 <= target_hair_w):
            hair_roi = resized_hair[fg_y1:fg_y2, fg_x1:fg_x2]
            mask_roi = resized_mask[fg_y1:fg_y2, fg_x1:fg_x2] / 255.0
            mask_roi = np.expand_dims(mask_roi, axis=-1)
            bg_roi = blended[bg_y1:bg_y2, bg_x1:bg_x2]
            blended[bg_y1:bg_y2, bg_x1:bg_x2] = ((1.0 - mask_roi) * bg_roi + mask_roi * hair_roi).astype(np.uint8)
            return blended, True
        return image_cv, False
    except Exception as e: 
        print(f"Hair Engine Error: {e}")
        return image_cv, False

# =========================================================
# [5] بيانات ومطابقة المخزن الذكي (Smart Database Mapper)
# =========================================================

inventory_df = pd.DataFrame()
links_df = pd.DataFrame()

LIVE_INVENTORY_URL = "https://www.royalelchim.app/acc/pscr3_1.aspx"
LOGIN_URL = "https://www.royalelchim.app/acc/login.aspx"
ACC_USERNAME = os.environ.get("ACC_USERNAME", "")
ACC_PASSWORD = os.environ.get("ACC_PASSWORD", "")

# دالة مطابقة الأعمدة الذكية للتأكد من نجاح البحث مهما اختلفت مسميات الملف المرفوع
def get_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = [str(c).strip() for c in df.columns]
    mapping = {'name': None, 'barcode': None, 'price': None}
    
    # 1. مطابقة عمود الاسم
    for c in cols:
        if any(x in c.lower() for x in ['صنف', 'اسم', 'المنتج', 'product', 'name', 'item']):
            mapping['name'] = c
            break
    if not mapping['name'] and len(cols) > 0:
        mapping['name'] = cols[0] # الوضع الافتراضي كأول عمود بالجدول
        
    # 2. مطابقة عمود الباركود
    for c in cols:
        if any(x in c.lower() for x in ['باركود', 'كود', 'الكود', 'barcode', 'code']) and c != mapping['name']:
            mapping['barcode'] = c
            break
            
    # 3. مطابقة عمود السعر الأساسي
    for c in cols:
        if any(x in c.lower() for x in ['سعر', 'price', 'كارت', 'قيمة']) and c != mapping['name']:
            mapping['price'] = c
            break
    return mapping

def fetch_live_data() -> bool:
    global inventory_df
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new") 
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        try:
            if ACC_USERNAME and ACC_PASSWORD:
                driver.get(LOGIN_URL)
                WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
                text_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='text'], input[type='email'], input:not([type])")
                if text_inputs: text_inputs[0].send_keys(ACC_USERNAME)
                pass_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
                if pass_inputs: pass_inputs[0].send_keys(ACC_PASSWORD)
                driver.find_element(By.CSS_SELECTOR, "input[type='submit'], button[type='submit'], input[value='دخول'], input[value='Login']").click()
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[value='المخازن']")))

            inventory_btn = driver.find_element(By.CSS_SELECTOR, "input[value='المخازن']")
            driver.execute_script("arguments[0].click();", inventory_btn)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='checkbox']")))
            
            # محاكاة تفعيل المخازن الستة وعلامة الصفر
            for cb in driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']"):
                if not cb.is_selected(): driver.execute_script("arguments[0].click();", cb)
                    
            show_btn = None
            for selector in ["input[value='عرض']", "input[value='OK']", "input[type='submit']", "button[value='عرض']", "button[value='OK']"]:
                try:
                    show_btn = driver.find_element(By.CSS_SELECTOR, selector)
                    if show_btn: break
                except: continue
                
            if not show_btn:
                for btn in driver.find_elements(By.TAG_NAME, "button"):
                    if btn.text and ("عرض" in btn.text or "OK" in btn.text):
                        show_btn = btn
                        break
            
            if show_btn:
                driver.execute_script("arguments[0].click();", show_btn)
                time.sleep(4) 
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            cards = soup.find_all('div', class_=lambda x: x and 'card-n' in x)
            if not cards: return False
                
            data = []
            for card in cards:
                try:
                    name_tag = card.find('a', class_=lambda x: x and 'grdi1' in x)
                    name = name_tag.text.strip() if name_tag else "بدون اسم"
                    qty = 0.0
                    qty_div = card.find('div', class_=lambda x: x and 'grdilqp' in x)
                    if qty_div:
                        spans = qty_div.find_all('span')
                        if spans:
                            try: qty = float(spans[0].text.strip())
                            except ValueError: pass
                            
                    price1 = 0.0
                    price_div = card.find('div', class_=lambda x: x and 'grdilpp' in x)
                    if price_div:
                        spans = price_div.find_all('span')
                        if spans:
                            try: price1 = float(spans[0].text.strip())
                            except ValueError: pass
                            
                    data.append({'الصنف': name, 'الباركود': '', 'سعر1 كارت': price1, 'اونلاين': qty})
                except Exception: continue

            if data:
                inventory_df = pd.DataFrame(data)
                return True
        finally: driver.quit()
    except Exception: return False
    return False

def load_data_in_memory():
    global inventory_df, links_df
    # 1. إعطاء الأولوية لملف الرفع اليدوي (آخر رفع قام به الأدمن)
    if os.path.exists("last_inventory_uploaded.csv"):
        try:
            inventory_df = pd.read_csv("last_inventory_uploaded.csv").fillna(0)
            print("✅ تم تحميل بيانات المخزن من آخر رفع يدوي بنجاح.")
        except: pass
    
    # 2. محاولة السحب الآلي إذا لم نجد ملف يدوي
    if inventory_df.empty:
        is_live_synced = fetch_live_data()
        if not is_live_synced:
            # البحث عن أي ملف يحتوي اسمه على 'last' في المستودع (مثيل: last.xls - Sheet1 (4).csv)
            files = os.listdir('.')
            inv_f = next((f for f in files if 'last' in f.lower() and f.endswith('.csv') and f != "last_inventory_uploaded.csv"), None)
            if inv_f: 
                inventory_df = pd.read_csv(inv_f).fillna(0)
                print(f"✅ تم تحميل بيانات المخزن من الملف المحلي: {inv_f}")
                
    # 3. التحميل الذكي لملف الروابط ( links_df ) لتفادي اختفاء الروابط المفقودة
    files = os.listdir('.')
    # الكود يبحث الآن بذكاء عن أي ملف يحتوي اسمه على 'final' أو 'combined' أو 'database' لتفادي فقدان الروابط
    links_f = next((f for f in files if any(x in f.lower() for x in ['final', 'combined', 'database', 'link']) and f.endswith('.csv')), None)
    if links_f:
        try:
            links_df = pd.read_csv(links_f).fillna("")
            print(f"✅ تم تحميل ملف الروابط المحلي بذكاء من: {links_f}")
        except Exception as e:
            print(f"⚠️ خطأ أثناء تحميل ملف الروابط: {e}")

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
# 👑 [6] الذكاء الاصطناعي - Royal Mind V4 Enterprise
# =========================================================

raw_env_keys = os.environ.get("GOOGLE_API_KEYS", "") + " " + os.environ.get("GOOGLE_API_KEY", "") + " " + os.environ.get("GEMINI_API_KEY", "")
SYSTEM_API_KEYS = list(set(re.findall(r'AIza[a-zA-Z0-9_-]+', raw_env_keys)))

def get_brand_catalog() -> str:
    return """
    مجموعة عطورنا الحصرية (ROYAL ELCHIM BRAND):
    1. Royal Black: شرقي خشبي.
    2. Royal Shine: فاكهي زهري.
    3. Royal Shadow: غامق دخاني.
    4. Royal Glow: فاكهي زهري دافئ.
    5. Royal Purpose: شرقي فاخر.
    """

PERSONAS = {
    "makeup_artist": (
        "أنتِ Royal Elchim Beauty Director.\n"
        "أنتِ لستِ مجرد خبيرة مكياج، أنتِ محللة ملامح وجه احترافية.\n"
        "عند الرد، اتبعي هذا التسلسل بدقة:\n"
        "1. ANALYZE: استخرجي نقاط القوة والمناطق التي يمكن إبرازها.\n"
        "2. SCORE: قيّمي من 100.\n"
        "3. EXPLAIN: اشرحي التأثير البصري المتوقع.\n"
        "4. RECOMMEND: قدمي 3 خيارات (Natural, Luxury, Celebrity).\n"
        "5. SELL: اختمي بتوصية Royal Elchim."
    ),
    "hair_stylist": (
        "أنتِ Royal Elchim Global Hair Architect.\n"
        "1. ANALYZE: حللي التناسب بين الوجه والقصة.\n"
        "2. SCORE: قيّمي اللون والقصة من 100.\n"
        "3. EXPLAIN: اشرحي لماذا يبدو التعديل أفضل.\n"
        "4. RECOMMEND: استنتجي أفضل القصات.\n"
        "5. SELL: اختمي بتوصية احترافية."
    ),
    "perfumer": (
        "أنتِ Royal Elchim Master Perfumer.\n"
        "1. ANALYZE: حللي شخصية العميل.\n"
        "2. SCORE: اختاري أقرب عطر جاهز.\n"
        "3. EXPLAIN: اشرحي سبب اختيار النوتة العطرية.\n"
        "4. RECOMMEND: ابتكري تركيبة حصرية (إجمالي 15ج، مثبت 3ج، كحول 32مل).\n"
        "5. SELL: اختمي برسالة فاخرة."
    ),
    "forever_friend": (
        "أنتِ Royal Mind. صديقة ذكية وراقية جداً.\n"
        "حللي المشاعر، قدمي الدعم المناسب بلغة طبيعية غير روبوتية، واختمي دائماً بسؤال ذكي."
    ),
    "greeter_new": ("أنتِ Royal Concierge. رحبي بالعميل الجديد، واشرحي أقسام النادي الـ VIP."),
    "greeter_returning": ("أنتِ Royal Concierge. استخرجي بيانات العميل من الذاكرة ورحبي به بشخصية دافئة وفاخرة."),
    "skin_doctor": (
        "أنتِ Royal Elchim Skin Doctor. طبيبة بشرة وخبيرة صحة خلوية.\n"
        "1. ANALYZE: تشخيص دقيق للون البشرة، المسام، الملمس، والمشاكل الحالية.\n"
        "2. SCORE: تقييم صحة البشرة ونضارتها من 100.\n"
        "3. EXPLAIN: التفسير العلمي المبسط لأسباب هذه المشاكل (إرهاق، أكسدة، جفاف).\n"
        "4. RECOMMEND: بروتوكول علاجي يومي وأسبوعي طبيعي وتجميلي.\n"
        "5. SELL: وصفة بمنتجات العناية بالبشرة المتاحة في Royal Elchim."
    ),
    "fashion_stylist": (
        "أنتِ Royal Elchim Fashion Stylist. منسقة أزياء عالمية.\n"
        "1. ANALYZE: تحليل شكل الجسم، الطول، الانحناءات، وطبيعة المناسبة أو الأسلوب المفضل.\n"
        "2. SCORE: تقييم الإطلالة المقترحة ومدى ملاءمتها للشخصية من 100.\n"
        "3. EXPLAIN: شرح سبب نجاح هذا التنسيق في إخفاء العيوب وإبراز الجمال.\n"
        "4. RECOMMEND: تنسيقات ملابس، أقمشة، وإكسسوارات معينة.\n"
        "5. SELL: ربط ستايل الملابس بالمكياج والعطر المثالي من Royal Elchim لإكمال اللوحة الجمالية."
    ),
    "color_analyst": (
        "أنتِ Royal Elchim Color Analyst. خبيرة تحليل الألوان الموسمية.\n"
        "1. ANALYZE: تحليل الألوان الموسمية للعميل (شتاء، ربيع، صيف، خريف) بناءً على البشرة والعيون.\n"
        "2. SCORE: تقييم نسبة توافق الألوان التي يرتديها حالياً مع طبيعته.\n"
        "3. EXPLAIN: شرح عميق لتأثير الألوان الصحيحة على إشراقة وجهه وإخفاء الهالات.\n"
        "4. RECOMMEND: تحديد باليت الألوان المثالية (Color Palette) لملابسه ومكياجه.\n"
        "5. SELL: توجيه العميل لاختيار درجات الصبغات والمكياج المتوفرة حصرياً لدينا."
    ),
    "luxury_sales": (
        "أنتِ Royal Elchim Luxury Sales Expert. خبيرة في سيكولوجية البيع الفاخر.\n"
        "1. ANALYZE: تحليل الاحتياجات الخفية للعميل والدوافع النفسية للشراء.\n"
        "2. SCORE: تقييم القيمة الشرائية ورفع الوعي بقيمة المنتج الفاخر.\n"
        "3. EXPLAIN: إيضاح القيمة المضافة، والتجربة الحصرية، وجودة الخامات لمنتجاتنا.\n"
        "4. RECOMMEND: اقتراح باقات شاملة ومنتجات مكملة (Cross-selling/Up-selling).\n"
        "5. SELL: إغلاق الصفقة بأسلوب ملكي راقٍ يجعل الشراء متعة وليس تكلفة."
    ),
    "image_consultant": (
        "أنتِ Royal Elchim Image Consultant. خبيرة بناء وتطوير الصورة الكلية (Image Consulting).\n"
        "1. ANALYZE: تحليل الكاريزما، الحضور البصري، والانطباع الأول الذي يتركه العميل.\n"
        "2. SCORE: تقييم قوة الحضور والتأثير المجتمعي والمهني من 100.\n"
        "3. EXPLAIN: تحليل لغة الجسد وتأثير المظهر العام على ثقة العميل بنفسه ونظرة الآخرين له.\n"
        "4. RECOMMEND: خطة Makeover شاملة (تعديلات شاملة للشعر، المكياج، العطر، وأسلوب التحدث).\n"
        "5. SELL: تسويق باقة التغيير الشامل من Royal Elchim كاستثمار في الذات."
    )
}

def robust_gen(contents, enable_search=False):
    if not SYSTEM_API_KEYS: return "⚠️ نظام الذكاء الاصطناعي معطل حالياً."
    for model in ["gemini-2.5-flash", "gemini-1.5-flash"]:
        for key in SYSTEM_API_KEYS:
            try:
                client = genai.Client(api_key=key)
                config = types.GenerateContentConfig(tools=[{"google_search": {}}]) if enable_search else None
                if isinstance(contents, list) and not enable_search:
                    res = client.models.generate_content(model=model, contents=contents)
                else:
                    res = client.models.generate_content(model=model, contents=contents, config=config)
                if res and res.text: return res.text
            except Exception as e: continue
    raise HTTPException(status_code=503, detail="خوادم الذكاء الاصطناعي لا تستجيب.")

# =========================================================
# [الـ API والبوابات]
# =========================================================

class CSVUploadPayload(BaseModel):
    csv_data: str

@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"): return FileResponse("index.html")
    return {"status": "error", "message": "ملف الواجهة index.html غير موجود بجانب ملف البايثون"}

@app.get("/logo.jpg")
async def serve_logo():
    for f in os.listdir('.'):
        if f.lower() in ["logo.jpg", "logo.jpeg", "logo.png", "logo.webp"]: return FileResponse(f)
    raise HTTPException(status_code=404, detail="ملف اللوجو غير موجود")

@app.get("/api/sync_inventory")
async def api_sync_inventory():
    success = fetch_live_data()
    if success: return {"status": "success", "message": "تم تحديث المخزن من النظام المحاسبي الحي بنجاح!"}
    return {"status": "warning", "message": "فشل جلب البيانات الحية بالمتصفح الآلي. سيتم استخدام الملفات المحلية."}

@app.post("/api/upload_inventory")
async def upload_inventory(payload: CSVUploadPayload):
    global inventory_df
    try:
        df = pd.read_csv(io.StringIO(payload.csv_data)).fillna(0)
        if df.empty:
            return {"status": "error", "message": "الملف المرفوع فارغ أو غير صالح."}
            
        inventory_df = df
        with open("last_inventory_uploaded.csv", "w", encoding="utf-8") as f:
            f.write(payload.csv_data)
            
        return {"status": "success", "message": f"تم استلام الجرد بنجاح! ({len(inventory_df)} صنف نشط)"}
    except Exception as e:
        return {"status": "error", "message": f"عذراً، صيغة الملف غير مدعومة: {str(e)}"}

@app.post("/api/consult")
async def consult(payload: dict):
    try:
        phone = payload.get('phone', 'مجهول')
        persona_key = payload.get('persona', 'skin_doctor')
        client_msg = payload.get('client_message', '')
        expert_persona = PERSONAS.get(persona_key, PERSONAS['skin_doctor'])
        
        prompt = f"{expert_persona}\nالذاكرة التراكمية وسجلات العميل السابقة: {get_history_from_db(phone)}\nالاستفسار الحالي: {client_msg}"
        res = robust_gen([prompt])
        
        save_to_db(phone, f"استشارة: {persona_key}", client_msg, res)
        return {"status": "success", "diagnosis": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/face")
async def face_login(payload: dict):
    if users_collection is None: raise HTTPException(status_code=500, detail="Database Error")
    try:
        img_data = base64.b64decode(payload["selfie"].split(",")[1])
        img_cv = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        current_sig = extract_face_signature(img_cv)
        if not current_sig: return {"status": "error", "message": "لم نتمكن من التقاط ملامح الوجه بوضوح."}
        
        all_users = list(users_collection.find({"face_signature": {"$exists": True}}))
        valid_users = [u for u in all_users if len(u.get("face_signature", [])) == len(current_sig)]
        
        if valid_users:
            db_sigs = np.array([u["face_signature"] for u in valid_users])
            curr_sig_arr = np.array(current_sig)
            distances = np.linalg.norm(db_sigs - curr_sig_arr, axis=1)
            best_idx = np.argmin(distances)
            min_dist = distances[best_idx]
            max_conf = max(0.0, 1.0 - (min_dist / 0.40)) * 100
            
            if max_conf >= 50.0:
                best_match = valid_users[best_idx]
                updated_sig = (np.array(best_match["face_signature"]) * 0.75 + curr_sig_arr * 0.25).tolist()
                users_collection.update_one({"phone": best_match["phone"]}, {"$set": {"face_signature": updated_sig}})
                welcome_msg = robust_gen([f"{PERSONAS['greeter_returning']}\nبناءً على ذاكرته: {get_history_from_db(best_match['phone'])}"])
                save_to_db(best_match['phone'], "Face ID", "دخول ذكي", welcome_msg)
                return {"status": "success", "is_new": False, "phone": best_match["phone"], "greeting": welcome_msg}
                
        if "phone" not in payload or not payload["phone"]:
            return {"status": "needs_phone", "message": "وجه جديد أو زاوية تصوير مختلفة."}
            
        users_collection.update_one({"phone": payload["phone"]}, {"$set": {"face_signature": current_sig}}, upsert=True)
        welcome_msg = robust_gen([f"{PERSONAS['greeter_new']}"])
        save_to_db(payload["phone"], "Face ID", "تسجيل بصمة", welcome_msg)
        return {"status": "success", "is_new": True, "phone": payload["phone"], "greeting": welcome_msg}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

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
        if is_fixed: final += p1 * qty
        else: final += [p1, p1*0.9, p1*0.85, p1*0.8][tier-1] * qty
    return {"total": final, "tier": tier, "msg": f"تم تطبيق أسعار الكارت {tier}"}

@app.get("/api/search")
async def search(query: str):
    try:
        if inventory_df.empty: return {"status": "error", "message": "مصفوفة الجرد فارغة حالياً."}
        
        # استخدام المطابق الذكي لتفادي تغير أسماء الأعمدة في قاعدة البيانات
        cols = get_column_mapping(inventory_df)
        name_col = cols['name']
        barcode_col = cols['barcode']
        price_col = cols['price']
        
        name_mask = inventory_df[name_col].astype(str).str.contains(query, case=False, na=False) if name_col else pd.Series([False]*len(inventory_df))
        barcode_mask = inventory_df[barcode_col].astype(str).str.contains(query, case=False, na=False) if barcode_col else pd.Series([False]*len(inventory_df))
        
        res = inventory_df[name_mask | barcode_mask].head(15)
        data = []
        for _, row in res.iterrows():
            name = str(row.get(name_col, '')) if name_col else 'بدون اسم'
            barcode = str(row.get(barcode_col, '---')) if barcode_col else '---'
            
            luxor = get_qty_by_keyword(row, ['اللوتس', 'لوتس', 'اقصر', 'luxor'])
            marwa = get_qty_by_keyword(row, ['المروة', 'المروه', 'marwa'])
            hurgada = get_qty_by_keyword(row, ['الغردقة', 'الغردقه', 'hurgada'])
            online = get_qty_by_keyword(row, ['اونلاين', 'online'])
            
            if any(kw in name.lower() for kw in ["زيت", "جرام", "تركيب", "كحول", "مثبت"]): luxor = 0
            
            p1 = clean_qty_value(row.get(price_col, 0)) if price_col else 0.0
            link = f"https://www.royalelchim.app/search?q={urllib.parse.quote(name)}"
            if not links_df.empty:
                match = links_df[links_df['name'].astype(str).str.contains(name, case=False, na=False)]
                if not match.empty: link = match.iloc[0]['item_page_link']
            data.append({
                "plain_name": name, "name": name, "price": p1, "barcode": barcode, "link": link, 
                "price1": p1, "price2": p1*0.9, "price3": p1*0.85, "price4": p1*0.8, 
                "is_fixed": "صافي" in name, 
                "online_qty": online, "luxor_qty": luxor, "marwa_qty": marwa, "hurgada_qty": hurgada, "show_link": online > 0
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
        
        if "hair" in mode: 
            dye_color_hex = payload.get("hair_color", "#000000")
            if dye_color_hex and dye_color_hex.startswith('#') and len(dye_color_hex) == 7:
                dye_rgb = tuple(int(dye_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            else: dye_rgb = None
            proc, _ = apply_royal_hair(img, p_img_cv, mode, dye_rgb)
        else: 
            proc, _ = apply_ar_effects(img, color, mode, texture=texture)
            
        _, buf = cv2.imencode('.webp', proc, [cv2.IMWRITE_WEBP_QUALITY, 80])
        res_b64 = f"data:image/webp;base64,{base64.b64encode(buf).decode()}"
        
        role = PERSONAS["hair_stylist"] if "hair" in mode else PERSONAS["makeup_artist"]
        user_comment = payload.get("user_comment", "")
        prompt = f"{role}\n"
        if user_comment: prompt += f"⚠️ تعليق العميل: '{user_comment}'.\n"
        prompt += f"العملية: {mode}، اللون: {payload.get('hair_color', texture)}\nالذاكرة التراكمية: {get_history_from_db(payload['phone'])}"
        
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
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/craft_perfume")
async def craft_perfume(payload: dict):
    try:
        phone = payload.get('phone', 'مجهول')
        client_msg = payload.get('client_message', '')
        available_oils = "- SAVVY Aventus (8 ج.م)\n- PARFUME OIL Rose Vanille (6 ج.م)\n"
        
        if not inventory_df.empty:
            cols = get_column_mapping(inventory_df)
            name_col = cols['name']
            price_col = cols['price']
            
            if name_col:
                target_keywords = 'PARFUME OIL|SAVVY'
                mask_oils = inventory_df[name_col].astype(str).str.contains(target_keywords, case=False, na=False)
                oils_df = inventory_df[mask_oils].head(80) 
                available_oils = ""
                for _, row in oils_df.iterrows():
                    name = str(row.get(name_col, ''))
                    price = clean_qty_value(row.get(price_col, 0)) if price_col else 0.0
                    if 0 < price < 150: available_oils += f"- {name} (السعر: {price} ج.م)\n"
        
        prompt = f"{PERSONAS['perfumer']}\nالكتالوج:\n{get_brand_catalog()}\nالزيوت:\n{available_oils}\nسجلات العميل: {get_history_from_db(phone)}\nالطلب: {client_msg}"
        res = robust_gen([prompt])
        
        button_html = ""
        if not links_df.empty:
            for _, row in links_df.iterrows():
                brand_name = str(row.get('name', ''))
                if brand_name and brand_name.lower() in res.lower() and len(brand_name) > 4:
                    link = row.get('item_page_link', '#')
                    button_html = f"\n<br><a href='{link}' target='_blank' class='btn gold-btn mt-3'><i class='fa-solid fa-cart-shopping'></i> اقتناء {brand_name} فوراً</a>"
                    break
        final_res = res + button_html
        save_to_db(phone, "تصميم عطر ملكي", client_msg, final_res)
        return {"status": "success", "answer": final_res}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vault")
async def vault(phone: str, request: Request):
    if vault_collection is None: return {"status": "error"}
    recs = list(vault_collection.find({"phone": str(phone).strip()}).sort("date", -1))
    base_url = str(request.base_url).rstrip('/')
    
    for r in recs:
        if r.get('selfie_url'):
            file_path = os.path.join(os.getcwd(), r['selfie_url'].lstrip('/'))
            if os.path.exists(file_path): r['selfie'] = base_url + r['selfie_url']
            else: r['selfie'] = None 

        if r.get('product_url'):
            file_path = os.path.join(os.getcwd(), r['product_url'].lstrip('/'))
            if os.path.exists(file_path): r['product'] = base_url + r['product_url']
            else: r['product'] = None 
                
    return {"status": "success", "data": [{"id": str(r['_id']), "type": r['type'], "userText": r.get('userText'), "aiText": r.get('aiText'), "selfie": r.get('selfie'), "product": r.get('product'), "date": r['date']} for r in recs]}

@app.delete("/api/vault/{record_id}/image")
async def delete_rec_image(record_id: str, phone: str):
    if vault_collection is not None:
        doc = vault_collection.find_one({"_id": ObjectId(record_id), "phone": str(phone).strip()})
        if doc:
            delete_file_from_disk(doc.get("selfie_url")); delete_file_from_disk(doc.get("product_url"))
            vault_collection.update_one({"_id": ObjectId(record_id)}, {"$set": {"selfie_url": None, "product_url": None}})
            return {"status": "success"}
    return {"status": "error", "message": "غير مصرح."}

@app.delete("/api/vault/{record_id}")
async def delete_rec(record_id: str, phone: str):
    if vault_collection is not None:
        doc = vault_collection.find_one({"_id": ObjectId(record_id), "phone": str(phone).strip()})
        if doc:
            delete_file_from_disk(doc.get("selfie_url")); delete_file_from_disk(doc.get("product_url"))
            vault_collection.delete_one({"_id": ObjectId(record_id)})
            return {"status": "success"}
    return {"status": "error", "message": "غير مصرح بالحذف."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
