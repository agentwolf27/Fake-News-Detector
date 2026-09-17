# Fake News Detector — the web app and API behind gunicorn on port 8080.
#
#   docker build -t fake-news-detector .
#   docker run --rm -p 8080:8080 fake-news-detector
#
# The models are not trained here: the committed pickles are the models (DECISIONS.md D1).
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
# Wheels only: nothing is compiled, so no compiler is needed in the image.
RUN pip install --only-binary=:all: -r requirements.txt

COPY app/app.py app/app.py
COPY utils/ utils/
COPY templates/index.html templates/index.html
COPY static/style.css static/app.js static/
COPY samples/ samples/
COPY model/fake_news_model.pkl model/vectorizer.pkl model/content_model.pkl model/content_vectorizer.pkl model/

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin detector
USER 10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --start-interval=1s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status == 200 else 1)"]

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "30", "--preload", "app.app:app"]
