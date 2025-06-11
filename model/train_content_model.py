import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
import joblib
import os
import sys
import re

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import clean_text

def remove_source_bias(text):
    """
    Remove source attribution to focus on content
    """
    # Remove common news source patterns
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
        r'^\w+ \([A-Z\s]+\) -',  # Generic location (SOURCE) - pattern
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

def load_and_process_data():
    """
    Load data with content-focused preprocessing
    """
    print("📚 LOADING DATA WITH CONTENT FOCUS")
    print("=" * 50)
    
    try:
        df = pd.read_csv('data/fake_or_real_news.csv')
        print(f"Original data shape: {df.shape}")
        
        # Enhanced preprocessing
        df['title_clean'] = df['title'].astype(str).apply(enhanced_clean_text)
        df['text_clean'] = df['text'].astype(str).apply(enhanced_clean_text)
        
        # Combine with content focus (less title weighting)
        df['content_text'] = df['title_clean'] + ' ' + df['text_clean']
        
        # Remove very short texts
        df = df[df['content_text'].str.len() > 30]
        
        print(f"After content processing: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def create_content_features(df):
    """
    Create features focused on content quality rather than source
    """
    print("\n🎯 CREATING CONTENT-FOCUSED FEATURES")
    print("=" * 50)
    
    # Content quality indicators
    df['word_count'] = df['content_text'].str.split().str.len()
    df['sentence_count'] = df['content_text'].str.count('\.')
    df['avg_word_length'] = df['content_text'].str.replace(' ', '').str.len() / df['word_count']
    
    # Writing quality indicators
    df['has_quotes'] = df['content_text'].str.contains('".*"', na=False)
    df['has_numbers'] = df['content_text'].str.contains(r'\d+', na=False)
    df['question_marks'] = df['content_text'].str.count('\?')
    df['exclamation_marks'] = df['content_text'].str.count('!')
    
    # Emotional language indicators
    df['caps_ratio'] = df['content_text'].str.count(r'[A-Z]') / df['content_text'].str.len()
    df['has_breaking'] = df['content_text'].str.contains('breaking', case=False, na=False)
    df['has_exclusive'] = df['content_text'].str.contains('exclusive|scoop', case=False, na=False)
    
    # Fake news indicators
    fake_keywords = ['miracle', 'amazing', 'shocking', 'unbelievable', 'secret', 'exposed', 'revealed']
    df['fake_keywords'] = df['content_text'].str.lower().str.count('|'.join(fake_keywords))
    
    print("Content features created:")
    print(f"  - Word count stats: {df['word_count'].describe()}")
    print(f"  - Quotes present: {df['has_quotes'].mean():.1%}")
    print(f"  - Numbers present: {df['has_numbers'].mean():.1%}")
    print(f"  - Fake keywords avg: {df['fake_keywords'].mean():.2f}")
    
    return df

def train_content_model(df):
    """
    Train a model focused on content rather than source
    """
    print("\n🤖 TRAINING CONTENT-FOCUSED MODEL")
    print("=" * 50)
    
    # Use content text without source bias
    X = df['content_text']
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Enhanced content-focused TF-IDF
    stop_words_extended = [
        'reuters', 'ap', 'associated', 'press', 'bloomberg', 'washington', 
        'new', 'york', 'london', 'sources', 'reported', 'according',
        'said', 'says', 'told', 'announced', 'confirmed', 'revealed',
        'exclusive', 'breaking', 'update', 'latest', 'just', 'now'
    ]
    
    vectorizer = TfidfVectorizer(
        max_features=15000,  # Increased features
        stop_words=stop_words_extended,
        ngram_range=(1, 3),  # Include trigrams
        min_df=3,
        max_df=0.8,
        sublinear_tf=True,
        use_idf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Content TF-IDF shape: {X_train_tfidf.shape}")
    
    # Train ensemble of content-focused models with calibration
    models = {
        'Content_LogReg': LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=1.0,
            class_weight='balanced',
            solver='liblinear'
        ),
        'Content_RF': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=15,
            min_samples_split=5
        ),
        'Content_SVM': SVC(
            random_state=42,
            probability=True,
            class_weight='balanced',
            kernel='linear'
        )
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation with probability calibration
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train and test
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        y_pred_proba = model.predict_proba(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Calculate uncertainty metrics
        uncertainty_threshold = 0.15
        uncertain_predictions = np.abs(y_pred_proba[:, 1] - 0.5) < uncertainty_threshold
        uncertainty_rate = np.mean(uncertain_predictions)
        
        results[name] = {
            'accuracy': accuracy,
            'auc': auc,
            'model': model,
            'uncertainty_rate': uncertainty_rate
        }
        
        print(f"Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        print(f"Uncertainty Rate: {uncertainty_rate:.2%}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
    
    print(f"\n🏆 Best content model accuracy: {best_score:.4f}")
    
    # Final evaluation with uncertainty analysis
    y_pred_final = best_model.predict(X_test_tfidf)
    y_pred_proba_final = best_model.predict_proba(X_test_tfidf)
    
    print(f"\nContent Model Classification Report:")
    print(classification_report(y_test, y_pred_final))
    
    # Analyze uncertain predictions
    uncertain_mask = np.abs(y_pred_proba_final[:, 1] - 0.5) < uncertainty_threshold
    uncertain_accuracy = accuracy_score(y_test[uncertain_mask], y_pred_final[uncertain_mask])
    print(f"\nUncertain Predictions Analysis:")
    print(f"Number of uncertain predictions: {np.sum(uncertain_mask)}")
    print(f"Accuracy on uncertain predictions: {uncertain_accuracy:.4f}")
    
    return best_model, vectorizer, best_score

def test_content_model(model, vectorizer):
    """
    Test the content-focused model
    """
    print("\n🧪 TESTING CONTENT-FOCUSED MODEL")
    print("=" * 50)
    
    test_cases = [
        # Should be REAL (good content, no sources)
        "Apple Inc. reported strong quarterly earnings today, beating analyst expectations with revenue growth of 15%. The technology giant saw increased sales across all product categories.",
        "Scientists at major universities have published peer-reviewed research showing climate change impacts on coral reefs. The study tracked 500 reefs over 10 years.",
        "The Federal Reserve announced a 0.25% interest rate increase following economic indicators showing steady growth and controlled inflation.",
        
        # Should be FAKE (poor content quality)
        "SHOCKING! This one simple trick will make you rich overnight! Doctors HATE this secret method!",
        "BREAKING: Aliens have been confirmed by government officials! They've been hiding among us for decades!",
        "Miracle cure discovered in grandmother's kitchen! Big pharma doesn't want you to know this ancient secret!",
        
        # Borderline cases
        "The earth is round and orbits the sun.",
        "Stock market volatility continues amid economic uncertainty.",
        "Breaking news: Major announcement expected from tech company."
    ]
    
    for i, text in enumerate(test_cases, 1):
        cleaned = enhanced_clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        
        fake_prob = probabilities[0]
        real_prob = probabilities[1]
        confidence = max(fake_prob, real_prob)
        
        is_uncertain = abs(fake_prob - real_prob) < 0.2
        
        status = "🟢 REAL" if prediction == "REAL" else "🔴 FAKE"
        if is_uncertain:
            status += " ⚠️  UNCERTAIN"
        
        print(f"{i:2d}. {status} - Conf: {confidence:.1%} (F:{fake_prob:.1%}, R:{real_prob:.1%})")
        print(f"    {text[:80]}...")
        print()

def save_content_model(model, vectorizer):
    """
    Save the content-focused model
    """
    # Save as content-specific model
    joblib.dump(model, 'model/content_model.pkl')
    joblib.dump(vectorizer, 'model/content_vectorizer.pkl')
    
    print("\n💾 Content-focused model saved as content_model.pkl")

def main():
    """
    Train content-focused model
    """
    print("🎯 CONTENT-FOCUSED FAKE NEWS DETECTION")
    print("=" * 60)
    
    # Load and process data
    df = load_and_process_data()
    if df is None:
        return
    
    # Create content features
    df = create_content_features(df)
    
    # Train content model
    model, vectorizer, accuracy = train_content_model(df)
    
    # Test content model
    test_content_model(model, vectorizer)
    
    # Save content model
    save_content_model(model, vectorizer)
    
    print("=" * 60)
    print(f"🎉 CONTENT MODEL COMPLETED! Accuracy: {accuracy:.4f}")
    print("🎯 This model focuses on content quality rather than source attribution")

if __name__ == "__main__":
    main() 
    main() 