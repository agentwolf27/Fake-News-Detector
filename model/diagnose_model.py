import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import clean_text

def load_model_components():
    """Load model and vectorizer"""
    try:
        model = joblib.load('model/fake_news_model.pkl')
        vectorizer = joblib.load('model/vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

def analyze_training_data():
    """Analyze what the model actually learned"""
    print("🔍 TRAINING DATA ANALYSIS")
    print("=" * 50)
    
    # Load training data
    df = pd.read_csv('data/fake_or_real_news.csv')
    
    # Analyze real vs fake patterns
    real_texts = df[df['label'] == 'REAL']['text'].str.lower()
    fake_texts = df[df['label'] == 'FAKE']['text'].str.lower()
    
    print("Key patterns the model learned:")
    print(f"1. Reuters in REAL news: {real_texts.str.contains('reuters').mean():.1%}")
    print(f"   Reuters in FAKE news: {fake_texts.str.contains('reuters').mean():.1%}")
    
    print(f"2. AP in REAL news: {real_texts.str.contains(' ap ').mean():.1%}")
    print(f"   AP in FAKE news: {fake_texts.str.contains(' ap ').mean():.1%}")
    
    print(f"3. Washington in REAL news: {real_texts.str.contains('washington').mean():.1%}")
    print(f"   Washington in FAKE news: {fake_texts.str.contains('washington').mean():.1%}")
    
    print(f"4. Average length REAL: {real_texts.str.len().mean():.0f} chars")
    print(f"   Average length FAKE: {fake_texts.str.len().mean():.0f} chars")
    
    # Show some real examples
    print("\nSample REAL news patterns:")
    for i, text in enumerate(real_texts.head(3)):
        print(f"  {i+1}. {text[:100]}...")
    
    print("\nSample FAKE news patterns:")
    for i, text in enumerate(fake_texts.head(3)):
        print(f"  {i+1}. {text[:100]}...")

def test_model_systematically():
    """Test model with systematic examples"""
    print("\n🧪 SYSTEMATIC MODEL TESTING")
    print("=" * 50)
    
    model, vectorizer = load_model_components()
    if model is None:
        print("❌ Model not found!")
        return
    
    # Test cases designed to understand bias
    test_groups = {
        "Professional News Format": [
            "WASHINGTON (Reuters) - Apple Inc. reported strong quarterly earnings today.",
            "NEW YORK (AP) - The Federal Reserve announced new monetary policy changes.",
            "LONDON (Bloomberg) - Stock markets rose following positive economic data."
        ],
        "Simple News Facts": [
            "Apple Inc. reported quarterly earnings today.",
            "The Federal Reserve announced new policies.",
            "Stock markets rose today."
        ],
        "Obvious Fake News": [
            "BREAKING: Aliens land in Times Square demanding pizza!",
            "Miracle cure for cancer found in grandmother's kitchen!",
            "Scientists prove the earth is actually flat!"
        ],
        "Short Statements": [
            "The earth is round.",
            "The earth is flat.",
            "Apple is a technology company."
        ]
    }
    
    for group_name, texts in test_groups.items():
        print(f"\n📋 {group_name}:")
        for i, text in enumerate(texts, 1):
            # Clean and predict
            cleaned = clean_text(text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]
            
            fake_prob = probabilities[0]
            real_prob = probabilities[1]
            
            status = "🟢 REAL" if prediction == "REAL" else "🔴 FAKE"
            print(f"  {i}. {status} (F:{fake_prob:.1%}, R:{real_prob:.1%}) - {text[:60]}...")

def analyze_feature_importance():
    """Analyze what features the model considers important"""
    print("\n🎯 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    
    model, vectorizer = load_model_components()
    if model is None:
        return
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # For Random Forest, get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 20 most important features:")
        for i, (feature, importance) in enumerate(feature_importance[:20]):
            print(f"  {i+1:2d}. {feature:20s} ({importance:.4f})")
    
    # For Logistic Regression, get coefficients
    elif hasattr(model, 'coef_'):
        coef = model.coef_[0]
        
        # Features that strongly indicate REAL news (negative coefficients)
        real_indicators = list(zip(feature_names, coef))
        real_indicators.sort(key=lambda x: x[1])
        
        print("Top 10 features indicating REAL news:")
        for i, (feature, coef) in enumerate(real_indicators[:10]):
            print(f"  {i+1:2d}. {feature:20s} ({coef:.4f})")
        
        print("\nTop 10 features indicating FAKE news:")
        fake_indicators = sorted(real_indicators, key=lambda x: x[1], reverse=True)
        for i, (feature, coef) in enumerate(fake_indicators[:10]):
            print(f"  {i+1:2d}. {feature:20s} ({coef:.4f})")

def recommend_improvements():
    """Recommend specific improvements"""
    print("\n💡 RECOMMENDATIONS FOR IMPROVEMENT")
    print("=" * 50)
    
    print("🔧 Issues identified:")
    print("  1. Model over-relies on source indicators (Reuters, AP)")
    print("  2. Short texts without source attribution get labeled as FAKE")
    print("  3. Model learned format patterns rather than content patterns")
    print("  4. Bias toward classifying non-professional format as FAKE")
    
    print("\n🚀 Suggested improvements:")
    print("  1. Add more diverse real news sources without Reuters/AP")
    print("  2. Include legitimate social media news posts in training")
    print("  3. Focus on content patterns rather than format")
    print("  4. Add uncertainty thresholds for borderline cases")
    print("  5. Use ensemble methods with different feature sets")
    
    print("\n📋 Next steps:")
    print("  1. Augment dataset with diverse real news sources")
    print("  2. Create balanced training data without source bias")
    print("  3. Implement content-based features (sentiment, topics)")
    print("  4. Add confidence intervals to predictions")

def main():
    """Run complete diagnosis"""
    print("🏥 FAKE NEWS MODEL DIAGNOSIS")
    print("=" * 60)
    
    analyze_training_data()
    test_model_systematically()
    analyze_feature_importance()
    recommend_improvements()
    
    print("\n" + "=" * 60)
    print("✅ Diagnosis complete! Check recommendations above.")

if __name__ == "__main__":
    main() 