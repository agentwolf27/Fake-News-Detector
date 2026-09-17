"""Text cleaning and scoring for the served models.

`read(text)` scores a text once with every loaded model and works out `input_check`.
`predict_ensemble(text)` turns that into the response the app has always returned; its
arithmetic is kept exactly as it was at 23207ce, so `tests/golden.json` still matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from utils.models import ModelPair, get_models

FORMAT_WEIGHT = 0.6
CONTENT_WEIGHT = 0.4
UNCERTAINTY_THRESHOLD = 0.15

# DECISIONS D2: every loaded model within this many points of its empty-text score.
NO_SIGNAL_POINTS = 3.0
# DECISIONS D3: refused when every loaded model's vocabulary coverage is below this.
COVERAGE_FLOOR = 0.5
# Largest text accepted, in UTF-8 bytes, on the API and on the page.
MAX_TEXT_BYTES = 256 * 1024

_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"http\S+|www\S+|https\S+", flags=re.MULTILINE)
_NON_LETTER = re.compile(r"[^a-zA-Z\s]")
_SPACES = re.compile(r"\s+")
# A whitespace token that is only a link: with a scheme, starting www., or host.tld/path.
_LINK = re.compile(r"(?:https?://\S+|www\.\S+|[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+/\S*)", re.IGNORECASE)
# Characters Python's str.split() and a browser's word count disagree on; both count them as spaces.
_EXTRA_SPACES = {cp: " " for cp in [*range(0x1C, 0x20), 0x85, 0xFEFF]}

_SOURCE_PATTERNS = [
    r"\(Reuters\)",
    r"\(AP\)",
    r"\(Associated Press\)",
    r"\(Bloomberg\)",
    r"WASHINGTON \(Reuters\)",
    r"NEW YORK \(AP\)",
    r"LONDON \(Bloomberg\)",
    r"Reuters -",
    r"AP -",
    r"Bloomberg -",
    r"^\w+ \([A-Z\s]+\) -",
]


class ModelUnavailable(Exception):
    """The format model did not load, so nothing can be scored."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _strip_tags(text: str) -> str:
    # Same result as re.sub(r'<[^>]+>', '', text), in linear time: a tag can only end at a '>',
    # so nothing after the last '>' can match, and every '<' before it finds its '>' at once.
    end = text.rfind(">")
    if end < 0:
        return text
    return _TAG.sub("", text[: end + 1]) + text[end + 1 :]


def clean_text(text) -> str:
    """Lowercase, strip HTML and URLs, keep letters and single spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _strip_tags(text)
    text = _URL.sub("", text)
    text = _NON_LETTER.sub("", text)
    return _SPACES.sub(" ", text).strip()


def remove_source_bias(text: str) -> str:
    """Remove agency names and datelines so the content model reads only the writing."""
    for pattern in _SOURCE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def enhanced_clean_text(text) -> str:
    """What the content model reads: source attribution removed, then cleaned."""
    if not isinstance(text, str):
        return ""
    return clean_text(remove_source_bias(text))


def word_count(text: str) -> int:
    return len(text.translate(_EXTRA_SPACES).split())


def is_link_only(text: str) -> bool:
    tokens = text.translate(_EXTRA_SPACES).split()
    return bool(tokens) and all(_LINK.fullmatch(t) for t in tokens)


@dataclass(frozen=True)
class ModelReading:
    key: str
    fake: float  # raw probabilities, never rounded here
    real: float
    label: str
    known_features: int
    coverage: float | None
    points_from_empty: float
    weight: float


@dataclass(frozen=True)
class Reading:
    text: str
    cleaned: str
    format: ModelReading
    content: ModelReading | None
    status: str  # ok | url_only | no_known_words | no_signal
    word_count: int

    @property
    def models(self) -> list[ModelReading]:
        return [r for r in (self.format, self.content) if r is not None]

    @property
    def empty(self) -> bool:
        """True when cleaning leaves nothing — the API answers 422."""
        return self.cleaned == ""

    def input_check(self) -> dict:
        def per_model(value):
            return {
                "format_model": value(self.format),
                "content_model": value(self.content) if self.content is not None else None,
            }

        return {
            "status": self.status,
            "word_count": self.word_count,
            "known_features": per_model(lambda r: r.known_features),
            "coverage": per_model(lambda r: None if r.coverage is None else round(r.coverage, 2)),
            "points_from_empty": per_model(lambda r: round(r.points_from_empty, 2)),
        }


def _read_pair(pair: ModelPair, text: str) -> ModelReading:
    fake, real, label, matrix = pair.proba(text)
    return ModelReading(
        key=pair.key,
        fake=fake,
        real=real,
        label=label,
        known_features=int(matrix.nnz),
        coverage=pair.coverage(text),
        points_from_empty=abs(float(fake) - pair.empty_fake) * 100,
        weight=pair.weight,
    )


def _status(text: str, readings: list[ModelReading]) -> str:
    if is_link_only(text):
        return "url_only"
    coverages = [r.coverage for r in readings if r.coverage is not None]
    if not coverages or all(c < COVERAGE_FLOOR for c in coverages):
        return "no_known_words"
    if all(r.points_from_empty <= NO_SIGNAL_POINTS for r in readings):
        return "no_signal"
    return "ok"


def read(text: str) -> Reading:
    """Score `text` once with every loaded model. Raises ModelUnavailable."""
    models = get_models()
    if not models.ready:
        raise ModelUnavailable(models.format.failure or "format model not loaded")
    cleaned = clean_text(text)
    fmt = _read_pair(models.format, cleaned)
    content = _read_pair(models.content, enhanced_clean_text(text)) if models.content.loaded else None
    readings = [r for r in (fmt, content) if r is not None]
    return Reading(
        text=text,
        cleaned=cleaned,
        format=fmt,
        content=content,
        status=_status(text, readings),
        word_count=word_count(text),
    )


def blend(reading: Reading) -> tuple[float, float]:
    """Raw blended (fake, real), with the same arithmetic the response uses."""
    f, c = reading.format, reading.content
    if c is None:
        return f.fake, f.real
    return (FORMAT_WEIGHT * f.fake + CONTENT_WEIGHT * c.fake, FORMAT_WEIGHT * f.real + CONTENT_WEIGHT * c.real)


def confidence_level(confidence: float) -> str:
    if confidence > 0.8:
        return "HIGH"
    if confidence > 0.65:
        return "MEDIUM"
    if confidence > 0.55:
        return "LOW"
    return "VERY_LOW"


def ensemble_result(reading: Reading) -> dict:
    """The response fields as 23207ce computed them. Only valid when not `reading.empty`."""
    text = reading.text
    f, c = reading.format, reading.content
    format_fake_prob, format_real_prob = f.fake, f.real
    format_prediction = f.label
    content_prediction = c.label if c is not None else None

    ensemble_fake_prob, ensemble_real_prob = blend(reading)
    if content_prediction is not None:
        final_prediction = "FAKE" if ensemble_fake_prob > ensemble_real_prob else "REAL"
        confidence = max(ensemble_fake_prob, ensemble_real_prob)
        models_agree = format_prediction == content_prediction
        is_uncertain = abs(ensemble_fake_prob - ensemble_real_prob) < UNCERTAINTY_THRESHOLD or not models_agree
    else:
        final_prediction = format_prediction
        confidence = max(ensemble_fake_prob, ensemble_real_prob)
        models_agree = True
        is_uncertain = abs(format_fake_prob - format_real_prob) < UNCERTAINTY_THRESHOLD

    return {
        "text": text[:200] + "..." if len(text) > 200 else text,
        "prediction": final_prediction,
        "confidence": round(float(confidence) * 100, 2),
        "confidence_level": confidence_level(confidence),
        "is_uncertain": bool(is_uncertain),
        "models_agree": bool(models_agree),
        "format_prediction": format_prediction,
        "content_prediction": content_prediction,
        "ensemble_fake_prob": round(float(ensemble_fake_prob) * 100, 2),
        "ensemble_real_prob": round(float(ensemble_real_prob) * 100, 2),
        "format_fake_prob": round(float(format_fake_prob) * 100, 2),
        "format_real_prob": round(float(format_real_prob) * 100, 2),
        "content_fake_prob": round(float(c.fake) * 100, 2) if content_prediction else None,
        "content_real_prob": round(float(c.real) * 100, 2) if content_prediction else None,
    }


def predict_ensemble(text: str) -> dict:
    """Both models, blended. Returns {"error": ...} instead of raising, as it always has."""
    try:
        reading = read(text)
    except ModelUnavailable:
        return {"error": "Format model not found"}
    if reading.empty:
        return {"error": "Empty text"}
    return ensemble_result(reading)


def get_prediction_details(text: str) -> dict:
    """Kept for the scripts that import it; same as predict_ensemble."""
    return predict_ensemble(text)


def load_model_and_vectorizer():
    """The loaded format model and vectorizer, or (None, None)."""
    pair = get_models().format
    return (pair.model, pair.vectorizer) if pair.loaded else (None, None)


def load_content_model():
    """The loaded content model and vectorizer, or (None, None)."""
    pair = get_models().content
    return (pair.model, pair.vectorizer) if pair.loaded else (None, None)


def predict_news(text: str) -> tuple:
    """Format model only: (label, confidence of that label)."""
    pair = get_models().format
    if not pair.loaded:
        return "Error: Model not found", 0.0
    cleaned = clean_text(text)
    if not cleaned:
        return "Error: Empty text", 0.0
    fake, real, label, _ = pair.proba(cleaned)
    return label, fake if label == "FAKE" else real


def predict_news_detailed(text: str) -> dict:
    """Format model only, with both class probabilities."""
    pair = get_models().format
    if not pair.loaded:
        return {"error": "Model not found"}
    cleaned = clean_text(text)
    if not cleaned:
        return {"error": "Empty text"}
    fake, real, label, _ = pair.proba(cleaned)
    return {
        "text": text[:200] + "..." if len(text) > 200 else text,
        "prediction": label,
        "confidence": round(max(fake, real) * 100, 2),
        "fake_probability": round(fake * 100, 2),
        "real_probability": round(real * 100, 2),
        "is_uncertain": bool(abs(fake - real) < 0.1),
    }
