import os
import logging
import threading
from typing import Optional
from flask import Flask, request, jsonify, render_template_string

# Flask app (Elastic Beanstalk Procfile expects "application:application")
application = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve artifact paths relative to this file; allow env overrides (empty env won't override)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH") or os.path.join(BASE_DIR, "model", "basic_classifier.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH") or os.path.join(BASE_DIR, "model", "count_vectorizer.pkl")

# Log resolved paths
logger.info("CWD: %s", os.getcwd())
logger.info("Resolved MODEL_PATH: %s", MODEL_PATH)
logger.info("Resolved VECTORIZER_PATH: %s", VECTORIZER_PATH)

# Global variables for loaded artifacts
_loaded_model: Optional[object] = None
_vectorizer: Optional[object] = None
_artifact_lock = threading.Lock()

# Artifact loading
def _load_artifacts_once() -> None:
    """Lazily load model and vectorizer once per process."""
    global _loaded_model, _vectorizer
    if _loaded_model is not None and _vectorizer is not None:
        return
    with _artifact_lock:
        if _loaded_model is None or _vectorizer is None:
            import pickle
            logger.info("Loading artifacts...")
            with open(MODEL_PATH, "rb") as mf:
                _loaded_model = pickle.load(mf)
            with open(VECTORIZER_PATH, "rb") as vf:
                _vectorizer = pickle.load(vf)
            logger.info("Artifacts loaded.")

# Inference function
def _predict_text(message: str) -> str:
    """Run inference and return the predicted class as a string label."""
    _load_artifacts_once()
    X = _vectorizer.transform([message])
    pred = _loaded_model.predict(X)
    # pred[0] could be a numpy scalar; normalize to native str
    val = pred[0]
    val_py = val.item() if hasattr(val, "item") else val
    return str(val_py)

# Eager load artifacts in a background thread at startup
def _eager_load_background():
    try:
        _load_artifacts_once()
    except Exception as e:
        # Log and continue; app remains healthy and will lazy-load on first request
        logger.warning("Background eager load failed: %s", e, exc_info=True)


# Non-blocking eager load at startup
threading.Thread(target=_eager_load_background, daemon=True).start()

DEMO_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Text Classifier Demo</title>
    <body>
    <style>
            h1 {
                font-size: 48px;
                font-weight: 700;
                margin-bottom: 20px;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                max-width: 1100px;
                margin: 0 auto;
                padding: 40px 20px;
                background-color: #f5f5f5;
            }
            .info {
                color: #666;
                margin-bottom: 30px;
                font-size: 16px;
            }
            label {
                display: block;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 10px;
            }
            textarea {
                width: 100%;
                min-height: 200px;
                padding: 15px;
                font-size: 16px;
                font-family: inherit;
                border: 1px solid #ddd;
                border-radius: 8px;
                box-sizing: border-box;
                resize: vertical;
            }
            button {
                margin-top: 20px;
                padding: 12px 30px;
                font-size: 16px;
                font-weight: 600;
                background-color: #e0e0e0;
                border: none;
                border-radius: 8px;
                cursor: pointer;
            }
            button:hover {
                background-color: #d0d0d0;
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                font-size: 18px;
            }
        </style>
    </head>
    <body>
        <h1>Text Classifier Demo</h1>
        <div class="info">
            Model loaded: {{ model_loaded }} — Model path: {{ model_path }}
        </div>
        <form method="POST" action="/predict-form">
            <label for="message">Enter text to classify:</label>
            <textarea id="message" name="message" placeholder="Type your text here...">{{ request.form.get('message', '') }}</textarea>
            <button type="submit">Predict</button>
        </form>
        {% if prediction %}
        <div class="result">
            Prediction: {{ prediction }}
        </div>
        {% endif %}
        {% if error %}
        <div class="error">
            Error: {{ error }}
        </div>
        {% endif %}
    </body>
</html>
"""

# Routes
@application.get("/")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(_loaded_model is not None and _vectorizer is not None),
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH,
    }), 200

# Demo page rendering endpoint
@application.get("/demo")
def demo():
    return render_template_string(
        DEMO_HTML,
        model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
        model_path=MODEL_PATH,
        prediction=None,
        error=None,
    )

# Form submission endpoint for demo page
@application.post("/predict-form")
def predict_form():
    message = (request.form.get("message") or "").strip()
    if not message:
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Field 'message' is required and must be non-empty.",
        ), 400
    try:
        label = _predict_text(message)
        return render_template_string(
            DEMO_HTML,
            model_loaded=True,
            model_path=MODEL_PATH,
            prediction=label,
            error=None,
        )
    except FileNotFoundError:
        return render_template_string(
            DEMO_HTML,
            model_loaded=False,
            model_path=MODEL_PATH,
            prediction=None,
            error="Model artifacts not found on server.",
        ), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Inference failed.",
        ), 500

# JSON API endpoint for predictions
@application.post("/predict")
def predict_json():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Field 'message' is required and must be non-empty."}), 400
    try:
        label = _predict_text(message)
        return jsonify({"label": label}), 200
    except FileNotFoundError:
        return jsonify({"error": "Model artifacts not found on server."}), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return jsonify({"error": "Inference failed."}), 500


if __name__ == "__main__":
    # Local dev run; in EB, Gunicorn (from Procfile) will host the app
    port = int(os.getenv("PORT", "8000"))
    application.run(host="0.0.0.0", port=port, debug=False)
