# 🌾 Crop Advisor — مستشار المحاصيل الزراعية

تطبيق Flask يستخدم نموذج Random Forest للتنبؤ بأنسب محصول زراعي بناءً على خصائص التربة والمناخ.

---

## 📁 هيكل المشروع

```
crop_advisor/
│
├── app.py                        # Flask app — نقطة الدخول الرئيسية
├── Crop_recommendation.csv       # بيانات التدريب
├── requirements.txt
│
├── model/
│   ├── __init__.py
│   ├── predictor.py              # CropPredictor class (train + predict)
│   ├── random_forest.joblib      # ← يُنشأ تلقائياً عند أول تشغيل
│   └── model_meta.json           # ← metadata + إحصائيات التدريب
│
├── templates/
│   └── index.html                # Jinja2 template
│
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── main.js
```

---

## ⚡ التشغيل

```bash
# 1. تثبيت المتطلبات
pip install -r requirements.txt

# 2. تشغيل الخادم
python app.py
```

ثم افتح المتصفح على: **http://localhost:5000**

> عند أول تشغيل، يتم تدريب النموذج تلقائياً وحفظه في `model/`

---

## 🔌 API Endpoints

| Method | Route            | الوصف                          |
|--------|------------------|-------------------------------|
| GET    | `/`              | الصفحة الرئيسية               |
| POST   | `/api/predict`   | التنبؤ بالمحصول (JSON)         |
| GET    | `/api/meta`      | بيانات النموذج والدقة          |
| GET    | `/api/health`    | health check                  |

### مثال — POST /api/predict

**Request:**
```json
{
  "N": 90, "P": 42, "K": 43,
  "temperature": 21, "humidity": 82,
  "ph": 6.5, "rainfall": 202
}
```

**Response:**
```json
{
  "crop": "rice",
  "crop_ar": "الأرز",
  "emoji": "🌾",
  "confidence": 99.5,
  "image_url": "https://...",
  "top_crops": [
    {"rank":1, "crop":"rice",  "confidence":99.5, "emoji":"🌾", "ar":"الأرز"},
    {"rank":2, "crop":"jute",  "confidence":0.3,  "emoji":"🌾", "ar":"الجوت"},
    {"rank":3, "crop":"maize", "confidence":0.1,  "emoji":"🌽", "ar":"الذرة"}
  ],
  "ideal_conditions": {
    "temperature": {"min":20,"max":27,"mean":23.7},
    "humidity":    {"min":80,"max":85,"mean":82.3},
    "ph":          {"min":5.0,"max":7.9,"mean":6.4},
    "rainfall":    {"min":183,"max":299,"mean":236}
  }
}
```

---

## 🧠 النموذج

| المعلومة | القيمة |
|---------|--------|
| Algorithm | Random Forest (200 trees) |
| Training samples | 2,200 |
| Classes | 22 نوع محصول |
| Test Accuracy | 99.55% |
| CV (5-Fold) | 99.38% ± 0.61% |

**أهم الـ Features:**
1. `rainfall` — معدل الأمطار
2. `humidity` — الرطوبة
3. `K` — البوتاسيوم
4. `temperature` — درجة الحرارة

---

## 🔄 إعادة التدريب

```python
from model.predictor import CropPredictor
p = CropPredictor(auto_train=False)
p.train(data_path="path/to/new_data.csv")
```

أو من command line:
```bash
python model/predictor.py
```
