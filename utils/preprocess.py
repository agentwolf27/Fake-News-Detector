import re
import string
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def clean_text(text):
    """
    Clean and preprocess text data
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_source_bias(text):
    """
    Remove source attribution to focus on content
    """
    patterns_to_remove = [
        r'\(Reuters\)',
        r'\(AP\)',
        r'\(Associated Press\)',
        r'\(Bloomberg\)',
        r'WASHINGTON \(Reuters\)',
        r'NEW YORK \(AP\)',
        r'LONDON \(Bloomberg\)',
        r'Reuters -',
        r'AP -',
        r'Bloomberg -',
        r'^\w+ \([A-Z\s]+\) -',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()

def enhanced_clean_text(text):
    """
    Enhanced text cleaning focused on content
    """
    if pd.isna(text):
        return ""
    
    # Remove source bias first
    text = remove_source_bias(text)
    
    # Standard cleaning
    text = clean_text(text)
    
    return text

def load_model_and_vectorizer():
    """
    Load the trained model and vectorizer (format-based)
    """
    try:
        model_path = os.path.join('model', 'fake_news_model.pkl')
        vectorizer_path = os.path.join('model', 'vectorizer.pkl')
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Format model files not found: {e}")
        return None, None

def load_content_model():
    """
    Load the content-focused model and vectorizer
    """
    try:
        model_path = os.path.join('model', 'content_model.pkl')
        vectorizer_path = os.path.join('model', 'content_vectorizer.pkl')
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Content model files not found: {e}")
        return None, None

def predict_news(text: str) -> tuple:
    """
    Predict if news article is fake or real (using format-based model)
    Returns: (label: str, probability: float)
    """
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        return "Error: Model not found", 0.0
    
    # Clean the input text
    cleaned_text = clean_text(text)
    
    if not cleaned_text.strip():
        return "Error: Empty text", 0.0
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Get prediction and probability
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    # Model classes are ['FAKE', 'REAL']
    fake_prob = probability[0]  # Probability of FAKE
    real_prob = probability[1]  # Probability of REAL
    
    # Return the confidence for the predicted class
    if prediction == 'FAKE':
        confidence = fake_prob
    else:
        confidence = real_prob
    
    return prediction, confidence

def predict_ensemble(text: str) -> dict:
    """
    Enhanced prediction using both format and content models
    """
    # Load both models
    format_model, format_vectorizer = load_model_and_vectorizer()
    content_model, content_vectorizer = load_content_model()
    
    if format_model is None or format_vectorizer is None:
        return {"error": "Format model not found"}
    
    # Format-based prediction
    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        return {"error": "Empty text"}
    
    format_vectorized = format_vectorizer.transform([cleaned_text])
    format_prediction = format_model.predict(format_vectorized)[0]
    format_proba = format_model.predict_proba(format_vectorized)[0]
    
    format_fake_prob = format_proba[0]
    format_real_prob = format_proba[1]
    
    # Content-based prediction (if available)
    content_prediction = None
    content_fake_prob = 0.5
    content_real_prob = 0.5
    
    if content_model is not None and content_vectorizer is not None:
        content_cleaned = enhanced_clean_text(text)
        content_vectorized = content_vectorizer.transform([content_cleaned])
        content_prediction = content_model.predict(content_vectorized)[0]
        content_proba = content_model.predict_proba(content_vectorized)[0]
        
        content_fake_prob = content_proba[0]
        content_real_prob = content_proba[1]
    
    # Ensemble prediction (weighted average)
    if content_prediction is not None:
        # Weight format model (good for source quality) and content model (good for content analysis)
        ensemble_fake_prob = (0.6 * format_fake_prob + 0.4 * content_fake_prob)
        ensemble_real_prob = (0.6 * format_real_prob + 0.4 * content_real_prob)
        
        # Final prediction
        final_prediction = "FAKE" if ensemble_fake_prob > ensemble_real_prob else "REAL"
        confidence = max(ensemble_fake_prob, ensemble_real_prob)
        
        # Check for model agreement
        models_agree = format_prediction == content_prediction
        
        # Uncertainty detection
        uncertainty_threshold = 0.15
        is_uncertain = (abs(ensemble_fake_prob - ensemble_real_prob) < uncertainty_threshold or 
                       not models_agree)
    else:
        # Fall back to format model only
        final_prediction = format_prediction
        ensemble_fake_prob = format_fake_prob
        ensemble_real_prob = format_real_prob
        confidence = max(ensemble_fake_prob, ensemble_real_prob)
        models_agree = True
        is_uncertain = abs(format_fake_prob - format_real_prob) < 0.15
    
    # Determine confidence level
    if confidence > 0.8:
        confidence_level = "HIGH"
    elif confidence > 0.65:
        confidence_level = "MEDIUM"
    elif confidence > 0.55:
        confidence_level = "LOW"
    else:
        confidence_level = "VERY_LOW"
    
    return {
        'text': text[:200] + '...' if len(text) > 200 else text,
        'prediction': final_prediction,
        'confidence': round(float(confidence) * 100, 2),
        'confidence_level': confidence_level,
        'is_uncertain': bool(is_uncertain),
        'models_agree': bool(models_agree),
        'format_prediction': format_prediction,
        'content_prediction': content_prediction,
        'ensemble_fake_prob': round(float(ensemble_fake_prob) * 100, 2),
        'ensemble_real_prob': round(float(ensemble_real_prob) * 100, 2),
        'format_fake_prob': round(float(format_fake_prob) * 100, 2),
        'format_real_prob': round(float(format_real_prob) * 100, 2),
        'content_fake_prob': round(float(content_fake_prob) * 100, 2) if content_prediction else None,
        'content_real_prob': round(float(content_real_prob) * 100, 2) if content_prediction else None
    }

def predict_news_detailed(text: str) -> dict:
    """
    Get detailed prediction with both class probabilities
    """
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        return {"error": "Model not found"}
    
    # Clean the input text
    cleaned_text = clean_text(text)
    
    if not cleaned_text.strip():
        return {"error": "Empty text"}
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Get prediction and probability
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    # Model classes are ['FAKE', 'REAL']
    fake_prob = probability[0]  # Probability of FAKE
    real_prob = probability[1]  # Probability of REAL
    
    return {
        'text': text[:200] + '...' if len(text) > 200 else text,
        'prediction': prediction,
        'confidence': round(max(fake_prob, real_prob) * 100, 2),
        'fake_probability': round(fake_prob * 100, 2),
        'real_probability': round(real_prob * 100, 2),
        'is_uncertain': abs(fake_prob - real_prob) < 0.1  # Flag uncertain predictions
    }

def get_prediction_details(text: str) -> dict:
    """
    Get enhanced prediction details using ensemble approach
    """
    return predict_ensemble(text) 