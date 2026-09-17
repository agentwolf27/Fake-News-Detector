"""Fake News Detector — the web page and the JSON API.

    .venv/bin/python app/app.py            # http://127.0.0.1:8080, debugger off
    PORT=8090 FLASK_DEBUG=1 .venv/bin/python app/app.py
    .venv/bin/gunicorn --bind 127.0.0.1:8080 --workers 2 --preload app.app:app
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sklearn  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

from utils import present  # noqa: E402
from utils.models import get_models  # noqa: E402
from utils.preprocess import (  # noqa: E402
    COVERAGE_FLOOR,
    MAX_TEXT_BYTES,
    NO_SIGNAL_POINTS,
    ModelUnavailable,
    ensemble_result,
    read,
    word_count,
)

MAX_BODY_BYTES = 1024 * 1024

app = Flask(__name__, template_folder=str(REPO_ROOT / "templates"), static_folder=str(REPO_ROOT / "static"))
app.config.update(MAX_CONTENT_LENGTH=MAX_BODY_BYTES, MAX_FORM_MEMORY_SIZE=MAX_BODY_BYTES)
log = logging.getLogger("fake_news_detector")

# Load the models once, at import, so gunicorn --preload shares them between workers.
get_models()


def _load_samples() -> list[dict]:
    index = REPO_ROOT / "samples" / "index.json"
    if not index.is_file():
        return []
    samples = []
    for entry in json.loads(index.read_text(encoding="utf-8")):
        body = json.loads((REPO_ROOT / "samples" / entry["file"]).read_text(encoding="utf-8"))
        samples.append({**entry, "text": body["text"]})
    return samples


SAMPLES = _load_samples()
SAMPLES_BY_ID = {s["id"]: s for s in SAMPLES}

DESCRIPTIONS = {
    "format_model": "Random forest over TF-IDF word patterns (one- to three-word phrases), including datelines and agency names",
    "content_model": "Logistic regression over TF-IDF words, read after agency names and datelines are removed",
    "ensemble": "Weighted blend: 60% format model, 40% content model",
}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra


def _wants_json() -> bool:
    return request.path.startswith("/api/") or request.path == "/health"


def _text_bytes(text: str) -> int:
    return len(text.encode("utf-8", "surrogatepass"))


# ---------------------------------------------------------------- JSON API


def _json_text() -> str:
    if not request.is_json:
        raise ApiError(415, "json_required", "Send the body as JSON with Content-Type: application/json.")
    try:
        data = request.get_json(silent=True)
    except RecursionError:
        data = None
    if not isinstance(data, dict):
        raise ApiError(400, "bad_json", 'The body must be a JSON object with a "text" field.')
    if not data or "text" not in data:
        raise ApiError(400, "text_missing", "No text provided")
    text = data["text"]
    if not isinstance(text, str):
        raise ApiError(400, "text_not_string", 'The "text" field must be a string.')
    size = _text_bytes(text)
    if size > MAX_TEXT_BYTES:
        raise ApiError(413, "too_large", f"The text is {size} bytes; the limit is {MAX_TEXT_BYTES} bytes.",
                       text_bytes=size, limit_bytes=MAX_TEXT_BYTES)
    if not text.strip():
        raise ApiError(400, "text_empty", "Empty text provided")
    return text


def _json_reading():
    text = _json_text()
    try:
        reading = read(text)
    except ModelUnavailable as exc:
        raise ApiError(503, "model_unavailable", "The model files did not load on this server.", reason=exc.reason)
    if reading.empty:
        if reading.status == "url_only":
            raise ApiError(422, "url_only", "The text is only a link; the detector reads text and does not fetch pages.",
                           input_check=reading.input_check())
        raise ApiError(422, "no_known_words", "Nothing in this text is left for the models to read.",
                       input_check=reading.input_check())
    return reading


@app.post("/api/predict")
def api_predict():
    reading = _json_reading()
    result = ensemble_result(reading)
    body = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "confidence_level": result["confidence_level"],
        "is_uncertain": result["is_uncertain"],
        "models_agree": result["models_agree"],
        "text": result["text"],
        "ensemble_probabilities": {"fake": result["ensemble_fake_prob"], "real": result["ensemble_real_prob"]},
        "model_breakdown": {
            "format_model": {
                "prediction": result["format_prediction"],
                "fake_prob": result["format_fake_prob"],
                "real_prob": result["format_real_prob"],
            }
        },
        "input_check": reading.input_check(),
    }
    if result.get("content_prediction") is not None:
        body["model_breakdown"]["content_model"] = {
            "prediction": result["content_prediction"],
            "fake_prob": result["content_fake_prob"],
            "real_prob": result["content_real_prob"],
        }
    return jsonify(body)


@app.post("/api/predict/simple")
def api_predict_simple():
    reading = _json_reading()
    result = ensemble_result(reading)
    return jsonify({
        "text": result["text"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "confidence_score": result["confidence"] / 100,
        "input_check": reading.input_check(),
    })


@app.get("/health")
def health():
    models = get_models()
    body = {
        "status": "healthy" if models.ready else "unavailable",
        "service": "fake-news-detector",
        "models": {
            key: "available" if pair.loaded else "not_available" for key, pair in models.pairs.items()
        },
        "sklearn_version": sklearn.__version__,
    }
    if not models.ready:
        body["reason"] = models.format.failure
    return jsonify(body), 200 if models.ready else 503


@app.get("/api/models")
def api_models():
    models = get_models()
    body = {}
    for key, pair in models.pairs.items():
        body[key] = {
            "description": DESCRIPTIONS[key],
            "available": all(f.exists for f in pair.files),
            "loaded": pair.loaded,
            "type": pair.model_type if pair.loaded else None,
            "n_features": pair.n_features if pair.loaded else None,
            "ngram_range": pair.ngram_range if pair.loaded else None,
            "weight": pair.weight,
            "empty_text_fake_prob": round(pair.empty_fake * 100, 2) if pair.loaded else None,
        }
    body["ensemble"] = {
        "description": DESCRIPTIONS["ensemble"],
        "available": models.ready,
        "loaded": models.ready,
        "weights": {key: pair.weight for key, pair in models.pairs.items()},
    }
    body["no_signal_points"] = NO_SIGNAL_POINTS
    body["coverage_floor"] = COVERAGE_FLOOR
    return jsonify(body)


# ---------------------------------------------------------------- page


def _render(view, *, text: str = "", status: int = 200, active_sample: str | None = None):
    return render_template(
        "index.html",
        view=view,
        text=text,
        count=present.words(word_count(text)),
        samples=SAMPLES,
        active_sample=active_sample,
        scope=present.SCOPE_LINE,
    ), status


def _score_page(text: str, *, active_sample: str | None = None):
    size = _text_bytes(text)
    if size > MAX_TEXT_BYTES:
        return _render(present.error("too_large", text_bytes=size, limit_bytes=MAX_TEXT_BYTES), status=413)
    if not text.strip():
        return _render(present.error("text_empty"), text=text, status=400)
    try:
        reading = read(text)
    except ModelUnavailable as exc:
        return _render(present.error("model_unavailable", detail=exc.reason), text=text, status=503)
    return _render(present.scored(reading, get_models()), text=text, active_sample=active_sample)


@app.get("/")
def page():
    sample_id = request.args.get("sample")
    if sample_id is None:
        return _render(present.first_load())
    sample = SAMPLES_BY_ID.get(sample_id)
    if sample is None:
        return _render(present.error("not_found", path=request.full_path.rstrip("?")), status=404)
    return _score_page(sample["text"], active_sample=sample_id)


@app.post("/")
def page_submit():
    return _score_page(request.form.get("text", ""))


# ---------------------------------------------------------------- errors

HTTP_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 413: "too_large", 415: "json_required"}


@app.errorhandler(ApiError)
def handle_api_error(exc: ApiError):
    return jsonify({"error": exc.message, "code": exc.code, **exc.extra}), exc.status


@app.errorhandler(HTTPException)
def handle_http_error(exc: HTTPException):
    status = exc.code or 500
    code = HTTP_CODES.get(status, f"http_{status}")
    if _wants_json():
        messages = {
            "not_found": "No such endpoint.",
            "method_not_allowed": "That method is not allowed on this endpoint.",
            "too_large": f"The request body is over {MAX_BODY_BYTES} bytes.",
            "json_required": "Send the body as JSON with Content-Type: application/json.",
        }
        return jsonify({"error": messages.get(code, exc.name), "code": code}), status
    if code == "not_found":
        return _render(present.error("not_found", path=request.path), status=404)
    return _render(present.error(code, detail=exc.name), status=status)


@app.errorhandler(Exception)
def handle_unexpected(exc: Exception):
    request_id = uuid.uuid4().hex
    log.exception("request %s failed", request_id)
    if _wants_json():
        return jsonify({"error": "Internal server error.", "code": "internal_error", "request_id": request_id}), 500
    return _render(present.error("internal_error", request_id=request_id), status=500)


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "") == "1"
    models = get_models()
    if not models.ready:
        print(f"Models not loaded: {models.format.failure}. The server will answer 503 until they are.")
    print(f"Fake News Detector on http://127.0.0.1:{port} (debugger {'on' if debug else 'off'})")
    app.run(host="127.0.0.1", port=port, debug=debug, use_reloader=debug)


if __name__ == "__main__":
    main()
