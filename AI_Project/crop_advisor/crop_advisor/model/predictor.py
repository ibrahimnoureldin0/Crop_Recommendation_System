"""
predictor.py
============
وحدة النموذج الرئيسية — تدريب وحفظ وتنبؤ المحصول الزراعي.

الاستخدام:
    from model.predictor import CropPredictor
    predictor = CropPredictor()
    result = predictor.predict(N=90, P=42, K=43,
                               temperature=21, humidity=82, ph=6.5, rainfall=202)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "random_forest.joblib"
META_PATH  = BASE_DIR / "model_meta.json"
DATA_PATH  = Path(__file__).parent.parent / "Crop_recommendation.csv"


# ── Arabic crop names & emoji ───────────────────────────────────────────────
CROP_INFO = {
    "apple":      {"ar": "التفاح",            "emoji": "🍎"},
    "banana":     {"ar": "الموز",             "emoji": "🍌"},
    "blackgram":  {"ar": "الماش الأسود",       "emoji": "🌿"},
    "chickpea":   {"ar": "الحمص",             "emoji": "🟡"},
    "coconut":    {"ar": "جوز الهند",          "emoji": "🥥"},
    "coffee":     {"ar": "القهوة",            "emoji": "☕"},
    "cotton":     {"ar": "القطن",             "emoji": "🌸"},
    "grapes":     {"ar": "العنب",             "emoji": "🍇"},
    "jute":       {"ar": "الجوت",             "emoji": "🌾"},
    "kidneybeans":{"ar": "الفاصوليا الحمراء",  "emoji": "🫘"},
    "lentil":     {"ar": "العدس",             "emoji": "🌾"},
    "maize":      {"ar": "الذرة",             "emoji": "🌽"},
    "mango":      {"ar": "المانجو",           "emoji": "🥭"},
    "mothbeans":  {"ar": "فاصوليا الموث",      "emoji": "🫘"},
    "mungbean":   {"ar": "الماش الأخضر",       "emoji": "🌱"},
    "muskmelon":  {"ar": "الشمام",            "emoji": "🍈"},
    "orange":     {"ar": "البرتقال",          "emoji": "🍊"},
    "papaya":     {"ar": "البابايا",          "emoji": "🍑"},
    "pigeonpeas": {"ar": "البازلاء الحمامية",  "emoji": "🌾"},
    "pomegranate":{"ar": "الرمان",            "emoji": "🍎"},
    "rice":       {"ar": "الأرز",             "emoji": "🌾"},
    "watermelon": {"ar": "البطيخ",            "emoji": "🍉"},
}

# ── Unsplash image URLs per crop ────────────────────────────────────────────
CROP_IMAGES = {
    "apple":      "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=300&h=300&fit=crop",
    "banana":     "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=300&h=300&fit=crop",
    "blackgram":  "https://images.unsplash.com/photo-1622206151226-18ca2c9ab4a1?w=300&h=300&fit=crop",
    "chickpea":   "https://images.unsplash.com/photo-1593050863699-7e5f5e4c6d6e?w=300&h=300&fit=crop",
    "coconut":    "https://images.unsplash.com/photo-1558642452-9d2a7deb7f62?w=300&h=300&fit=crop",
    "coffee":     "https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=300&h=300&fit=crop",
    "cotton":     "https://images.unsplash.com/photo-1598300042247-d088f8ab3a91?w=300&h=300&fit=crop",
    "grapes":     "https://images.unsplash.com/photo-1537640538966-79f369143f8f?w=300&h=300&fit=crop",
    "jute":       "https://images.unsplash.com/photo-1622206151226-18ca2c9ab4a1?w=300&h=300&fit=crop",
    "kidneybeans":"https://images.unsplash.com/photo-1525059696034-4967a8e1dca2?w=300&h=300&fit=crop",
    "lentil":     "https://images.unsplash.com/photo-1623428187969-5da2dcea5ebf?w=300&h=300&fit=crop",
    "maize":      "https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=300&h=300&fit=crop",
    "mango":      "https://images.unsplash.com/photo-1553279768-865429fa0078?w=300&h=300&fit=crop",
    "mothbeans":  "https://images.unsplash.com/photo-1525059696034-4967a8e1dca2?w=300&h=300&fit=crop",
    "mungbean":   "https://images.unsplash.com/photo-1622206151226-18ca2c9ab4a1?w=300&h=300&fit=crop",
    "muskmelon":  "https://images.unsplash.com/photo-1571194573770-73f8d35302c7?w=300&h=300&fit=crop",
    "orange":     "https://images.unsplash.com/photo-1547514701-42782101795e?w=300&h=300&fit=crop",
    "papaya":     "https://images.unsplash.com/photo-1526318896980-cf78c088247c?w=300&h=300&fit=crop",
    "pigeonpeas": "https://images.unsplash.com/photo-1623428187969-5da2dcea5ebf?w=300&h=300&fit=crop",
    "pomegranate":"https://images.unsplash.com/photo-1541344999736-83eca272f6fc?w=300&h=300&fit=crop",
    "rice":       "https://images.unsplash.com/photo-1536304929831-ee1ca9d44906?w=300&h=300&fit=crop",
    "watermelon": "https://images.unsplash.com/photo-1587049352846-4a222e784d38?w=300&h=300&fit=crop",
}

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# ═══════════════════════════════════════════════════════════════════════════
class CropPredictor:
    """
    واجهة موحدة للتدريب والتنبؤ.

    Attributes
    ----------
    model   : RandomForestClassifier المدرَّب
    encoder : LabelEncoder لتحويل الأرقام ← أسماء المحاصيل
    meta    : dict يحتوي على دقة النموذج وإحصائيات التدريب
    """

    def __init__(self, auto_train: bool = True):
        self.model   = None
        self.encoder = None
        self.meta    = {}

        if MODEL_PATH.exists() and META_PATH.exists():
            self._load()
        elif auto_train:
            print("[CropPredictor] No saved model found — training now...")
            self.train()
        else:
            raise FileNotFoundError("Model not found. Call .train() first.")

    # ── Training ────────────────────────────────────────────────────────────
    def train(self, data_path: str = None) -> dict:
        """
        تدريب نموذج Random Forest على بيانات المحاصيل.

        Parameters
        ----------
        data_path : مسار ملف CSV (اختياري — يستخدم DATA_PATH افتراضياً)

        Returns
        -------
        dict : تقرير التدريب (دقة Train/Test/CV)
        """
        csv = Path(data_path) if data_path else DATA_PATH
        if not csv.exists():
            raise FileNotFoundError(f"Dataset not found at: {csv}")

        df = pd.read_csv(csv)
        X  = df[FEATURE_COLS]
        y  = df["label"]

        # Encode labels
        self.encoder = LabelEncoder()
        y_enc = self.encoder.fit_transform(y)

        # Train / test split — stratified
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        # Model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Metrics
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        test_acc  = accuracy_score(y_test,  self.model.predict(X_test))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring="accuracy")

        # Feature importance
        fi = dict(zip(FEATURE_COLS, self.model.feature_importances_.tolist()))

        # Crop stats for UI hints
        crop_stats = {}
        for crop in df["label"].unique():
            sub = df[df["label"] == crop]
            crop_stats[crop] = {
                f: [round(float(sub[f].min()), 2),
                    round(float(sub[f].max()), 2),
                    round(float(sub[f].mean()), 2)]
                for f in FEATURE_COLS
            }

        self.meta = {
            "train_accuracy":    round(train_acc, 6),
            "test_accuracy":     round(test_acc, 6),
            "cv_mean":           round(float(cv_scores.mean()), 6),
            "cv_std":            round(float(cv_scores.std()), 6),
            "n_samples":         int(len(df)),
            "n_classes":         int(len(self.encoder.classes_)),
            "classes":           self.encoder.classes_.tolist(),
            "feature_importance": fi,
            "crop_stats":        crop_stats,
            "feature_ranges": {
                f: {
                    "min":  round(float(df[f].min()), 2),
                    "max":  round(float(df[f].max()), 2),
                    "mean": round(float(df[f].mean()), 2),
                }
                for f in FEATURE_COLS
            },
        }

        self._save()

        print(f"[CropPredictor] Training complete!")
        print(f"  Train Accuracy : {train_acc:.4f}")
        print(f"  Test  Accuracy : {test_acc:.4f}")
        print(f"  CV (5-Fold)    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return self.meta

    # ── Predict ─────────────────────────────────────────────────────────────
    def predict(
        self,
        N: float, P: float, K: float,
        temperature: float, humidity: float,
        ph: float, rainfall: float,
        top_n: int = 3,
    ) -> dict:
        """
        التنبؤ بأنسب محصول بناء على متغيرات التربة والمناخ.

        Parameters
        ----------
        N           : نسبة النيتروجين  (mg/kg)
        P           : نسبة الفوسفور   (mg/kg)
        K           : نسبة البوتاسيوم  (mg/kg)
        temperature : درجة الحرارة    (°C)
        humidity    : الرطوبة النسبية  (%)
        ph          : حموضة التربة    (3.5 – 10)
        rainfall    : الأمطار السنوية  (mm)
        top_n       : عدد التوصيات المطلوب إرجاعها

        Returns
        -------
        dict : نتيجة التنبؤ (crop, confidence, top_n, crop_info, ideal_conditions)
        """
        if self.model is None or self.encoder is None:
            raise RuntimeError("Model not loaded. Call .train() first.")

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        pred_idx  = self.model.predict(features)[0]
        proba     = self.model.predict_proba(features)[0]

        crop_name  = self.encoder.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx]) * 100

        # Top-N recommendations
        top_indices = np.argsort(proba)[::-1][:top_n]
        top_crops = [
            {
                "rank":       int(i + 1),
                "crop":       self.encoder.inverse_transform([idx])[0],
                "confidence": round(float(proba[idx]) * 100, 2),
                "emoji":      CROP_INFO.get(self.encoder.inverse_transform([idx])[0], {}).get("emoji", "🌱"),
                "ar":         CROP_INFO.get(self.encoder.inverse_transform([idx])[0], {}).get("ar", ""),
            }
            for i, idx in enumerate(top_indices)
        ]

        # Ideal conditions from training stats
        ideal = self.meta.get("crop_stats", {}).get(crop_name, {})

        return {
            "crop":       crop_name,
            "crop_ar":    CROP_INFO.get(crop_name, {}).get("ar", crop_name),
            "emoji":      CROP_INFO.get(crop_name, {}).get("emoji", "🌱"),
            "image_url":  CROP_IMAGES.get(crop_name, ""),
            "confidence": round(confidence, 2),
            "top_crops":  top_crops,
            "ideal_conditions": {
                f: {"min": ideal[f][0], "max": ideal[f][1], "mean": ideal[f][2]}
                for f in FEATURE_COLS
                if f in ideal
            },
            "input": {
                "N": N, "P": P, "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            },
        }

    # ── Persistence ─────────────────────────────────────────────────────────
    def _save(self):
        joblib.dump({"model": self.model, "encoder": self.encoder}, MODEL_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        print(f"[CropPredictor] Model saved → {MODEL_PATH}")

    def _load(self):
        bundle       = joblib.load(MODEL_PATH)
        self.model   = bundle["model"]
        self.encoder = bundle["encoder"]
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        print(f"[CropPredictor] Model loaded from {MODEL_PATH}")
        print(f"  Test Accuracy: {self.meta.get('test_accuracy', 'N/A')}")

    # ── Convenience ─────────────────────────────────────────────────────────
    @property
    def feature_ranges(self) -> dict:
        """نطاقات كل Feature للـ UI sliders."""
        return self.meta.get("feature_ranges", {})

    @property
    def crop_stats(self) -> dict:
        """إحصائيات كل محصول."""
        return self.meta.get("crop_stats", {})

    @property
    def accuracy(self) -> float:
        """دقة النموذج على بيانات الاختبار."""
        return self.meta.get("test_accuracy", 0.0)


# ── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = CropPredictor()
    print("\n── Test prediction (Rice conditions) ──")
    result = p.predict(N=90, P=42, K=43, temperature=21,
                       humidity=82, ph=6.5, rainfall=202)
    print(f"  Recommended crop : {result['crop']} {result['emoji']}")
    print(f"  Confidence       : {result['confidence']:.2f}%")
    print(f"  Top 3:")
    for t in result["top_crops"]:
        print(f"    {t['rank']}. {t['crop']:15s} {t['confidence']:.2f}%")
