# استخدام نسخة بايثون خفيفة
FROM python:3.9-slim

# تحديد مجلد العمل داخل السيرفر
WORKDIR /app

# تثبيت متطلبات النظام الأساسية لتشغيل OpenCV و MediaPipe بسلام
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كل ملفات المشروع
COPY . .

# فتح المنفذ الخاص بـ FastAPI
EXPOSE 8080

# أمر التشغيل الملكي
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
