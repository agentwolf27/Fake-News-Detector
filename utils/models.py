"""Load the two served models once, and say exactly what loaded.

The directory defaults to `model/` next to this repository and can be moved with
`FND_MODEL_DIR`. Each file is loaded independently; a failure is kept as the exception's class
name and the file's basename — never a path, never the message — so the API can report it.

A model pair counts as loaded only when both of its files load and it can score an empty
string. That empty-string score is also the baseline for the "too little to go on" rule
(DECISIONS D2).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "model"

# key -> (model file, vectorizer file, blend weight)
PAIRS: dict[str, tuple[str, str, float]] = {
    "format_model": ("fake_news_model.pkl", "vectorizer.pkl", 0.6),
    "content_model": ("content_model.pkl", "content_vectorizer.pkl", 0.4),
}


@dataclass(frozen=True)
class FileRecord:
    basename: str
    exists: bool
    error: str | None  # exception class name when loading failed


@dataclass(frozen=True)
class ModelPair:
    key: str
    weight: float
    files: tuple[FileRecord, FileRecord]
    model: object | None = None
    vectorizer: object | None = None
    fake_index: int | None = None
    real_index: int | None = None
    empty_fake: float | None = None  # raw FAKE probability for ""
    empty_real: float | None = None
    stop_words: frozenset = field(default_factory=frozenset)
    preprocessor: object | None = None
    tokenizer: object | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.empty_fake is not None

    @property
    def failure(self) -> str | None:
        """A short, path-free reason the pair is not loaded, or None."""
        if self.loaded:
            return None
        for rec in self.files:
            if not rec.exists:
                return f"{rec.basename} not found"
            if rec.error:
                return f"{rec.basename} failed to load ({rec.error})"
        return f"{self.files[0].basename} could not score an empty text"

    def proba(self, text: str):
        """Return (fake, real, label, matrix) for one already-cleaned text. Raw floats.

        The label is the class with the highest probability, first one on a tie — exactly what
        `predict` returns for both served models — so the forest only runs once.
        """
        matrix = self.vectorizer.transform([text])
        row = self.model.predict_proba(matrix)[0]
        label = str(self.model.classes_[int(row.argmax())])
        return row[self.fake_index], row[self.real_index], label, matrix

    def tokens(self, text: str) -> list[str]:
        """The vectorizer's own unigram tokens for `text`, stop words removed."""
        return [t for t in self.tokenizer(self.preprocessor(text)) if t not in self.stop_words]

    def coverage(self, text: str) -> float | None:
        toks = self.tokens(text)
        if not toks:
            return None
        vocab = self.vectorizer.vocabulary_
        return sum(1 for t in toks if t in vocab) / len(toks)

    @property
    def model_type(self) -> str | None:
        return type(self.model).__name__ if self.model is not None else None

    @property
    def n_features(self) -> int | None:
        return len(self.vectorizer.vocabulary_) if self.vectorizer is not None else None

    @property
    def ngram_range(self) -> list[int] | None:
        return list(self.vectorizer.ngram_range) if self.vectorizer is not None else None


@dataclass(frozen=True)
class Models:
    directory: str  # basename only, for messages
    pairs: dict[str, ModelPair]

    @property
    def format(self) -> ModelPair:
        return self.pairs["format_model"]

    @property
    def content(self) -> ModelPair:
        return self.pairs["content_model"]

    @property
    def ready(self) -> bool:
        """The service can score only when the format pair is loaded."""
        return self.format.loaded


def _load_file(directory: Path, basename: str):
    path = directory / basename
    if not path.is_file():
        return None, FileRecord(basename, False, None)
    try:
        return joblib.load(path), FileRecord(basename, True, None)
    except Exception as exc:  # a corrupt or incompatible pickle must not stop the server
        return None, FileRecord(basename, True, type(exc).__name__)


def _load_pair(directory: Path, key: str) -> ModelPair:
    model_file, vec_file, weight = PAIRS[key]
    model, model_rec = _load_file(directory, model_file)
    vec, vec_rec = _load_file(directory, vec_file)
    files = (model_rec, vec_rec)
    if model is None or vec is None:
        return ModelPair(key, weight, files)
    try:
        classes = [str(c) for c in model.classes_]
        fake_index, real_index = classes.index("FAKE"), classes.index("REAL")
        stops = vec.get_stop_words()
        pair = ModelPair(
            key,
            weight,
            files,
            model=model,
            vectorizer=vec,
            fake_index=fake_index,
            real_index=real_index,
            stop_words=frozenset(stops) if stops else frozenset(),
            preprocessor=vec.build_preprocessor(),
            tokenizer=vec.build_tokenizer(),
        )
        # Scoring "" once is the warm-up (the first transform sets an attribute on the
        # vectorizer) and the baseline for DECISIONS D2.
        fake, real, _, _ = pair.proba("")
    except Exception as exc:
        return ModelPair(key, weight, (FileRecord(model_file, True, type(exc).__name__), vec_rec))
    return replace(pair, empty_fake=float(fake), empty_real=float(real))


def load_models(directory: str | os.PathLike | None = None) -> Models:
    directory = Path(directory or os.environ.get("FND_MODEL_DIR") or DEFAULT_MODEL_DIR)
    return Models(directory.name, {key: _load_pair(directory, key) for key in PAIRS})


_current: Models | None = None


def get_models() -> Models:
    global _current
    if _current is None:
        _current = load_models()
    return _current


@contextmanager
def use_models(models: Models) -> Iterator[Models]:
    """Temporarily serve a different set of models (tests only)."""
    global _current
    previous = _current
    _current = models
    try:
        yield models
    finally:
        _current = previous
