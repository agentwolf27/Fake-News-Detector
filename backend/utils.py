import os
import joblib
import re

def clean_text(text):
    # Basic text cleaning
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'vectorizer.pkl')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].max()
    return {'prediction': pred, 'confidence': float(proba)} 