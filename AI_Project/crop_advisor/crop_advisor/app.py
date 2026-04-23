"""
app.py
======
تطبيق Flask الرئيسي — يربط النموذج بالواجهة الأمامية.

Routes
------
GET  /             → الصفحة الرئيسية
POST /api/predict  → تنبؤ JSON
GET  /api/meta     → بيانات النموذج (دقة، نطاقات، إحصائيات)
GET  /api/health   → health check
"""

from flask import Flask, render_template, request, jsonify
from model.predictor import CropPredictor
import traceback

# ── App init ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False   # دعم العربية في JSON

# تحميل / تدريب النموذج عند بدء الخادم
predictor = CropPredictor()


# ══════════════════════════════════════════════════════════════════════════════
# Pages
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """الصفحة الرئيسية — تمرر بيانات النموذج للـ template."""
    return render_template(
        "index.html",
        feature_ranges=predictor.feature_ranges,
        crop_stats=predictor.crop_stats,
        model_accuracy=f"{predictor.accuracy * 100:.2f}%",
        n_samples=predictor.meta.get("n_samples", 0),
        n_classes=predictor.meta.get("n_classes", 0),
        cv_mean=f"{predictor.meta.get('cv_mean', 0) * 100:.2f}%",
    )


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST /api/predict
    -----------------
    Body (JSON):
        { "N":90, "P":42, "K":43, "temperature":21,
          "humidity":82, "ph":6.5, "rainfall":202 }

    Response (JSON):
        { crop, crop_ar, emoji, image_url, confidence,
          top_crops, ideal_conditions, input }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Validate & parse
        required = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        missing  = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 422

        params = {k: float(data[k]) for k in required}

        # Run prediction
        result = predictor.predict(**params)
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 422
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/meta", methods=["GET"])
def api_meta():
    """
    GET /api/meta
    -------------
    يرجع بيانات النموذج: دقة، نطاقات الـ features، أهمية كل feature.
    """
    return jsonify({
        "test_accuracy":      predictor.meta.get("test_accuracy"),
        "train_accuracy":     predictor.meta.get("train_accuracy"),
        "cv_mean":            predictor.meta.get("cv_mean"),
        "cv_std":             predictor.meta.get("cv_std"),
        "n_samples":          predictor.meta.get("n_samples"),
        "n_classes":          predictor.meta.get("n_classes"),
        "classes":            predictor.meta.get("classes", []),
        "feature_importance": predictor.meta.get("feature_importance", {}),
        "feature_ranges":     predictor.feature_ranges,
    }), 200


@app.route("/api/health", methods=["GET"])
def api_health():
    """GET /api/health — للتأكد إن الخادم شغال."""
    return jsonify({
        "status":   "ok",
        "model":    "RandomForest",
        "accuracy": predictor.accuracy,
    }), 200


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
