import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
        
        # Check if dataset exists, if not try to download it
        if not os.path.exists(data_path):
            print("📥 Dataset not found. Attempting to download...")
            
            # Import and run the download script
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
        
    except FileNotFoundError:
        print("❌ Dataset not found. Please run 'python model/download_dataset.py' first.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the dataset
    """
    # Combine title and text for better prediction
    df['combined_text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    
    # Clean the combined text
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['cleaned_text'].str.len() > 0]
    
    print(f"Data after preprocessing. Shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df

def train_model(df):
    """
    Train the fake news detection model
    """
    # Features and target
    X = df['cleaned_text']
    y = df['label']
    
    # Split the data (check if stratification is possible)
    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    
    if min_class_count >= 2 and len(class_counts) >= 2:
        # Use stratification if we have at least 2 samples per class
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        # Simple random split if stratification isn't possible
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Fit vectorizer on training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    
    # Train Logistic Regression model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0,
        class_weight='balanced'
    )
    
    print("Training model...")
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, vectorizer, accuracy

def save_model(model, vectorizer):
    """
    Save the trained model and vectorizer
    """
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, 'model/fake_news_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    
    print("\nModel and vectorizer saved successfully!")

def main():
    """
    Main training pipeline
    """
    print("Starting Fake News Detection Model Training...")
    print("=" * 50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Train model
    model, vectorizer, accuracy = train_model(df)
    
    # Save model
    save_model(model, vectorizer)
    
    print("=" * 50)
    print(f"Training completed! Final accuracy: {accuracy:.4f}")
    
    # Test with sample predictions
    print("\nTesting with sample predictions:")
    print("-" * 30)
    
    sample_texts = [
        "Scientists discover new cure for cancer in breakthrough study",
        "Aliens spotted landing in downtown New York City",
        "Stock market reaches new record high amid economic growth"
    ]
    
    for text in sample_texts:
        prediction = model.predict(vectorizer.transform([clean_text(text)]))[0]
        proba = model.predict_proba(vectorizer.transform([clean_text(text)]))[0]
        confidence = max(proba) * 100
        
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {prediction} (Confidence: {confidence:.1f}%)")
        print()

if __name__ == "__main__":
    main() 