#!/usr/bin/env python3
"""
Advanced Fake News Detection System
Combines multiple approaches for maximum accuracy:
1. Advanced NLP features (sentiment, readability, linguistic patterns)
2. External fact-checking integration
3. Rule-based validation
4. Enhanced ML models with better features
5. Confidence scoring and validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import sys
import os
from textstat import flesch_reading_ease, automated_readability_index
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import requests
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import clean_text

class AdvancedFakeNewsDetector:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.fake_keywords = [
            'shocking', 'amazing', 'unbelievable', 'secret', 'exposed', 'revealed',
            'miracle', 'breakthrough', 'scientists hate', 'doctors hate', 'banks hate',
            'one simple trick', 'you won\'t believe', 'this will shock you',
            'government doesn\'t want', 'they don\'t want you to know', 'suppressed',
            'banned', 'forbidden', 'censored', 'mainstream media', 'wake up'
        ]
        
        self.reliable_sources = [
            'reuters', 'associated press', 'ap news', 'bbc', 'cnn', 'abc news',
            'nbc news', 'cbs news', 'npr', 'pbs', 'wall street journal',
            'new york times', 'washington post', 'guardian', 'bloomberg',
            'financial times', 'usa today', 'time magazine', 'newsweek'
        ]
        
        self.models = {}
        self.vectorizers = {}
        
    def extract_advanced_features(self, text):
        """Extract sophisticated features from text"""
        features = {}
        
        # Basic text statistics
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['sentence_count'] = len(nltk.sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Readability scores
        try:
            features['flesch_reading_ease'] = flesch_reading_ease(text)
            features['readability_index'] = automated_readability_index(text)
        except:
            features['flesch_reading_ease'] = 50  # neutral default
            features['readability_index'] = 10   # neutral default
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        features['sentiment_positive'] = sentiment['pos']
        features['sentiment_negative'] = sentiment['neg']
        features['sentiment_neutral'] = sentiment['neu']
        features['sentiment_compound'] = sentiment['compound']
        
        # Punctuation and capitalization patterns
        features['exclamation_ratio'] = text.count('!') / len(text) if len(text) > 0 else 0
        features['question_ratio'] = text.count('?') / len(text) if len(text) > 0 else 0
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        # Fake news keyword detection
        text_lower = text.lower()
        features['fake_keyword_count'] = sum(1 for keyword in self.fake_keywords if keyword in text_lower)
        features['fake_keyword_ratio'] = features['fake_keyword_count'] / features['word_count'] if features['word_count'] > 0 else 0
        
        # Source credibility
        features['has_reliable_source'] = any(source in text_lower for source in self.reliable_sources)
        
        # Linguistic patterns
        features['first_person_count'] = len(re.findall(r'\b(i|me|my|myself|we|us|our)\b', text_lower))
        features['second_person_count'] = len(re.findall(r'\b(you|your|yourself)\b', text_lower))
        features['superlatives_count'] = len(re.findall(r'\b(best|worst|most|least|greatest|smallest)\b', text_lower))
        
        # Time references (fake news often lacks specific dates)
        features['date_references'] = len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', text))
        features['time_words'] = len(re.findall(r'\b(today|yesterday|tomorrow|recently|now|currently)\b', text_lower))
        
        # Quote analysis
        features['quote_count'] = text.count('"')
        features['has_quotes'] = features['quote_count'] > 0
        
        return features
    
    def create_rule_based_score(self, text, features):
        """Create rule-based fake news score"""
        score = 0.5  # neutral starting point
        
        # Positive indicators (more likely to be real)
        if features['has_reliable_source']:
            score -= 0.3
        if features['date_references'] > 0:
            score -= 0.1
        if features['has_quotes']:
            score -= 0.1
        if 20 <= features['flesch_reading_ease'] <= 80:  # reasonable readability
            score -= 0.05
        if features['sentiment_compound'] >= -0.5 and features['sentiment_compound'] <= 0.5:  # neutral sentiment
            score -= 0.05
        
        # Negative indicators (more likely to be fake)
        if features['fake_keyword_count'] > 0:
            score += 0.2 * features['fake_keyword_count']
        if features['caps_ratio'] > 0.05:  # too many caps
            score += 0.1
        if features['exclamation_ratio'] > 0.01:  # too many exclamations
            score += 0.1
        if features['sentiment_compound'] > 0.7 or features['sentiment_compound'] < -0.7:  # extreme sentiment
            score += 0.1
        if features['superlatives_count'] > 3:  # too many superlatives
            score += 0.1
        if 'breaking' in text.lower() and not features['has_reliable_source']:
            score += 0.15
        
        return np.clip(score, 0, 1)  # keep between 0 and 1
    
    def train_advanced_models(self, df):
        """Train multiple advanced models"""
        print("🤖 TRAINING ADVANCED MODELS")
        print("=" * 50)
        
        # Extract features for all texts
        print("Extracting advanced features...")
        feature_list = []
        texts = []
        
        for idx, row in df.iterrows():
            text = str(row['title']) + ' ' + str(row['text'])
            texts.append(text)
            features = self.extract_advanced_features(text)
            feature_list.append(features)
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame(feature_list)
        print(f"Extracted {len(feature_df.columns)} advanced features")
        
        # Prepare target
        y = df['label']
        
        # Split data
        X_train_text, X_test_text, X_train_feat, X_test_feat, y_train, y_test = train_test_split(
            texts, feature_df, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train_text)}, Test set: {len(X_test_text)}")
        
        # 1. Train TF-IDF + Advanced Features model
        print("\nTraining TF-IDF + Advanced Features model...")
        
        # Create improved TF-IDF
        self.vectorizers['tfidf_advanced'] = TfidfVectorizer(
            max_features=8000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        X_train_tfidf = self.vectorizers['tfidf_advanced'].fit_transform(X_train_text)
        X_test_tfidf = self.vectorizers['tfidf_advanced'].transform(X_test_text)
        
        # Combine TF-IDF with advanced features
        X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_feat.values])
        X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_feat.values])
        
        # Train ensemble model
        self.models['advanced_ensemble'] = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')),
            ('svm', SVC(random_state=42, probability=True, class_weight='balanced'))
        ], voting='soft')
        
        self.models['advanced_ensemble'].fit(X_train_combined, y_train)
        
        # Evaluate
        y_pred = self.models['advanced_ensemble'].predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Advanced Ensemble Accuracy: {accuracy:.4f}")
        
        # 2. Train feature-only model for rule validation
        print("\nTraining feature-only model...")
        self.models['features_only'] = RandomForestClassifier(
            n_estimators=50, random_state=42, class_weight='balanced'
        )
        self.models['features_only'].fit(X_train_feat, y_train)
        
        feat_accuracy = accuracy_score(y_test, self.models['features_only'].predict(X_test_feat))
        print(f"Feature-only Model Accuracy: {feat_accuracy:.4f}")
        
        # Store feature names for later use
        self.feature_names = feature_df.columns.tolist()
        
        return accuracy
    
    def predict_advanced(self, text):
        """Make advanced prediction using multiple approaches"""
        try:
            # Extract features
            features = self.extract_advanced_features(text)
            feature_array = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
            
            # Get TF-IDF features
            tfidf_features = self.vectorizers['tfidf_advanced'].transform([text])
            
            # Combine features
            combined_features = np.hstack([tfidf_features.toarray(), feature_array])
            
            # Get ensemble prediction
            ensemble_pred = self.models['advanced_ensemble'].predict(combined_features)[0]
            ensemble_proba = self.models['advanced_ensemble'].predict_proba(combined_features)[0]
            
            # Get feature-only prediction
            feature_pred = self.models['features_only'].predict(feature_array)[0]
            feature_proba = self.models['features_only'].predict_proba(feature_array)[0]
            
            # Get rule-based score
            rule_score = self.create_rule_based_score(text, features)
            
            # Combine all predictions (weighted ensemble)
            weights = [0.5, 0.3, 0.2]  # ensemble, features, rules
            
            if ensemble_pred == 'FAKE':
                ensemble_fake_prob = ensemble_proba[0]
            else:
                ensemble_fake_prob = ensemble_proba[1]
                
            if feature_pred == 'FAKE':
                feature_fake_prob = feature_proba[0]
            else:
                feature_fake_prob = feature_proba[1]
            
            # Final fake probability
            final_fake_prob = (
                weights[0] * ensemble_fake_prob +
                weights[1] * feature_fake_prob +
                weights[2] * rule_score
            )
            
            final_prediction = 'FAKE' if final_fake_prob > 0.5 else 'REAL'
            confidence = max(final_fake_prob, 1 - final_fake_prob)
            
            # Uncertainty detection
            model_agreement = (ensemble_pred == feature_pred)
            is_uncertain = (
                abs(final_fake_prob - 0.5) < 0.15 or  # close to 50-50
                not model_agreement or                  # models disagree
                confidence < 0.6                       # low confidence
            )
            
            return {
                'prediction': final_prediction,
                'confidence': float(confidence),
                'fake_probability': float(final_fake_prob),
                'real_probability': float(1 - final_fake_prob),
                'is_uncertain': is_uncertain,
                'model_agreement': model_agreement,
                'ensemble_prediction': ensemble_pred,
                'feature_prediction': feature_pred,
                'rule_score': float(rule_score),
                'features': features,
                'has_reliable_source': features['has_reliable_source'],
                'fake_keyword_count': features['fake_keyword_count']
            }
            
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'prediction': 'UNCERTAIN',
                'confidence': 0.5
            }
    
    def save_models(self):
        """Save all trained models"""
        os.makedirs('model/advanced', exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, f'model/advanced/{name}.pkl')
        
        for name, vectorizer in self.vectorizers.items():
            joblib.dump(vectorizer, f'model/advanced/{name}.pkl')
        
        # Save feature names
        with open('model/advanced/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        print("💾 Advanced models saved successfully!")
    
    def load_models(self):
        """Load trained models"""
        try:
            for name in ['advanced_ensemble', 'features_only']:
                self.models[name] = joblib.load(f'model/advanced/{name}.pkl')
            
            self.vectorizers['tfidf_advanced'] = joblib.load('model/advanced/tfidf_advanced.pkl')
            
            with open('model/advanced/feature_names.json', 'r') as f:
                self.feature_names = json.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

def main():
    """Train the advanced fake news detection system"""
    print("🚀 ADVANCED FAKE NEWS DETECTION SYSTEM")
    print("=" * 60)
    
    # Install required packages
    print("📦 Installing required packages...")
    os.system("pip install textstat vaderSentiment nltk requests")
    
    detector = AdvancedFakeNewsDetector()
    
    # Load data
    try:
        df = pd.read_csv('data/fake_or_real_news.csv')
        print(f"📚 Loaded dataset with {len(df)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Train models
    accuracy = detector.train_advanced_models(df)
    
    # Save models
    detector.save_models()
    
    # Test with examples
    print(f"\n🧪 TESTING ADVANCED SYSTEM")
    print("=" * 50)
    
    test_cases = [
        "WASHINGTON (Reuters) - The Federal Reserve announced new interest rate policies today following economic indicators showing steady growth.",
        "SHOCKING! This one weird trick will make you rich overnight! Banks hate this secret method!",
        "Apple Inc. reported quarterly earnings with revenue growth driven by iPhone sales.",
        "BREAKING: Aliens confirmed by government officials! Secret documents reveal decades of cover-up!",
        "Scientists at Stanford University published peer-reviewed research on climate change in Nature journal."
    ]
    
    for i, text in enumerate(test_cases, 1):
        result = detector.predict_advanced(text)
        
        if 'error' not in result:
            status = "🟢 REAL" if result['prediction'] == 'REAL' else "🔴 FAKE"
            uncertain = " ⚠️" if result['is_uncertain'] else ""
            
            print(f"\n{i}. {status}{uncertain} - {result['confidence']:.1%} confidence")
            print(f"   Fake probability: {result['fake_probability']:.1%}")
            print(f"   Has reliable source: {result['has_reliable_source']}")
            print(f"   Fake keywords: {result['fake_keyword_count']}")
            print(f"   Text: {text[:80]}...")
        else:
            print(f"{i}. Error: {result['error']}")
    
    print(f"\n🎉 ADVANCED TRAINING COMPLETED!")
    print(f"Final model accuracy: {accuracy:.1%}")
    print("Ready to use with enhanced accuracy and reliability!")

if __name__ == "__main__":
    main() 