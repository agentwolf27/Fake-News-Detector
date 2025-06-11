import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import clean_text

def load_data():
    """
    Load the fake news dataset
    """
    try:
        data_path = os.path.join('data', 'fake_or_real_news.csv')
        
        if not os.path.exists(data_path):
            print("📥 Dataset not found. Attempting to download...")
            try:
                from download_dataset import main as download_main
                if not download_main():
                    print("❌ Failed to download dataset. Using fallback.")
                    return None
            except ImportError:
                print("❌ Dataset downloader not available.")
                return None
        
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def analyze_data(df):
    """
    Analyze the dataset for potential issues
    """
    print("\n📊 DATASET ANALYSIS")
    print("=" * 50)
    
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Check text lengths
    df['text_length'] = df['text'].str.len()
    print(f"\nText length statistics:")
    print(df.groupby('label')['text_length'].describe())
    
    # Check for common patterns
    real_texts = df[df['label'] == 'REAL']['text'].str.lower()
    fake_texts = df[df['label'] == 'FAKE']['text'].str.lower()
    
    print(f"\nReal news often contains 'Reuters': {real_texts.str.contains('reuters').mean():.1%}")
    print(f"Fake news often contains 'Reuters': {fake_texts.str.contains('reuters').mean():.1%}")
    
    print(f"Real news average length: {real_texts.str.len().mean():.0f} chars")
    print(f"Fake news average length: {fake_texts.str.len().mean():.0f} chars")

def preprocess_data_improved(df):
    """
    Enhanced preprocessing with better feature engineering
    """
    print("\n🔧 IMPROVED PREPROCESSING")
    print("=" * 50)
    
    # Combine title and text more strategically
    df['title_clean'] = df['title'].astype(str).apply(clean_text)
    df['text_clean'] = df['text'].astype(str).apply(clean_text)
    
    # Weight title more heavily (it's often more indicative)
    df['combined_text'] = df['title_clean'] + ' ' + df['title_clean'] + ' ' + df['text_clean']
    
    # Remove very short texts (likely incomplete)
    df = df[df['combined_text'].str.len() > 50]
    
    # Add feature engineering
    df['has_reuters'] = df['text'].str.contains('Reuters|AP |Associated Press', case=False, na=False)
    df['has_quotes'] = df['text'].str.contains('".*"', na=False)
    df['has_caps'] = df['text'].str.contains('[A-Z]{3,}', na=False)
    df['word_count'] = df['combined_text'].str.split().str.len()
    
    print(f"Data after improved preprocessing. Shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Show feature correlations
    print(f"\nFeature analysis:")
    print(f"Reuters mentions - Real: {df[df['label']=='REAL']['has_reuters'].mean():.1%}, Fake: {df[df['label']=='FAKE']['has_reuters'].mean():.1%}")
    print(f"Has quotes - Real: {df[df['label']=='REAL']['has_quotes'].mean():.1%}, Fake: {df[df['label']=='FAKE']['has_quotes'].mean():.1%}")
    
    return df

def train_improved_model(df):
    """
    Train an improved model with better features and hyperparameters
    """
    print("\n🤖 TRAINING IMPROVED MODEL")
    print("=" * 50)
    
    # Features and target
    X = df['combined_text']
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Improved TF-IDF with better parameters
    vectorizer = TfidfVectorizer(
        max_features=15000,  # Increased features
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams
        min_df=3,           # Higher minimum document frequency
        max_df=0.85,        # Lower maximum document frequency
        sublinear_tf=True,  # Apply sublinear scaling
        use_idf=True
    )
    
    # Fit vectorizer
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    
    # Try multiple models
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=2.0,
            class_weight='balanced',
            solver='liblinear'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=20,
            min_samples_split=5
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train and evaluate
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        y_pred_proba = model.predict_proba(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best model: {best_name} (Accuracy: {best_score:.4f})")
    
    # Final evaluation
    y_pred_final = best_model.predict(X_test_tfidf)
    print(f"\nFinal Classification Report:")
    print(classification_report(y_test, y_pred_final))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_final))
    
    return best_model, vectorizer, best_score

def test_model_thoroughly(model, vectorizer):
    """
    Thoroughly test the model with various examples
    """
    print("\n🧪 COMPREHENSIVE MODEL TESTING")
    print("=" * 50)
    
    test_cases = [
        # Professional news formats (should be REAL)
        "WASHINGTON (Reuters) - The Federal Reserve announced today that interest rates will remain unchanged following a two-day policy meeting.",
        "NEW YORK (AP) - Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales.",
        "LONDON (Bloomberg) - The Bank of England raised interest rates by 0.25 percentage points to combat rising inflation.",
        
        # Legitimate but simple news (should be REAL or uncertain)
        "The stock market closed higher today after positive economic data was released.",
        "Scientists at Stanford University published a new study on climate change in Nature journal.",
        
        # Conspiracy/fake patterns (should be FAKE)
        "BREAKING: Government officials reveal aliens have been living among us for decades!",
        "This one simple trick doctors don't want you to know will cure cancer overnight!",
        "The earth is actually flat and NASA has been lying to us for years!",
        
        # Borderline cases (might be uncertain)
        "The earth is round",
        "The earth is flat",
        "Breaking news: Major announcement expected"
    ]
    
    for i, text in enumerate(test_cases, 1):
        cleaned_text = clean_text(text)
        text_vectorized = vectorizer.transform([cleaned_text])
        
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        fake_prob = probabilities[0]
        real_prob = probabilities[1]
        confidence = max(fake_prob, real_prob)
        
        is_uncertain = abs(fake_prob - real_prob) < 0.15
        
        status = "🔴 FAKE" if prediction == "FAKE" else "🟢 REAL"
        if is_uncertain:
            status += " ⚠️  (UNCERTAIN)"
        
        print(f"{i:2d}. {status} - Conf: {confidence:.1%} (F:{fake_prob:.1%}, R:{real_prob:.1%})")
        print(f"    Text: {text[:80]}...")
        print()

def save_improved_model(model, vectorizer):
    """
    Save the improved model
    """
    # Backup old model
    if os.path.exists('model/fake_news_model.pkl'):
        os.rename('model/fake_news_model.pkl', 'model/fake_news_model_old.pkl')
    if os.path.exists('model/vectorizer.pkl'):
        os.rename('model/vectorizer.pkl', 'model/vectorizer_old.pkl')
    
    # Save new model
    joblib.dump(model, 'model/fake_news_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    
    print("\n💾 Improved model saved successfully!")
    print("    Old model backed up as *_old.pkl")

def main():
    """
    Main improved training pipeline
    """
    print("🚀 IMPROVED FAKE NEWS DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Load and analyze data
    df = load_data()
    if df is None:
        return
    
    analyze_data(df)
    
    # Enhanced preprocessing
    df = preprocess_data_improved(df)
    
    # Train improved model
    model, vectorizer, accuracy = train_improved_model(df)
    
    # Comprehensive testing
    test_model_thoroughly(model, vectorizer)
    
    # Save model
    save_improved_model(model, vectorizer)
    
    print("=" * 60)
    print(f"🎉 TRAINING COMPLETED! Final accuracy: {accuracy:.4f}")
    print("🔧 Model improvements:")
    print("   ✅ Better text preprocessing")
    print("   ✅ Enhanced TF-IDF parameters")
    print("   ✅ Multiple model comparison")
    print("   ✅ Comprehensive testing")

if __name__ == "__main__":
    main() 