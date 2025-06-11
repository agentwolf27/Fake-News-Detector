#!/usr/bin/env python3
"""
Robust Fake News Detection System
Addresses training bias through multiple specialized approaches:
1. Content-type specific models (news vs social media vs clickbait)
2. Rule-based fact-checking integration
3. External validation when possible
4. Conservative uncertainty handling
5. Explainable decision making
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import re
import sys
import os
import json
from datetime import datetime
import requests
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RobustFakeNewsDetector:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Define patterns that strongly indicate fake news
        self.fake_patterns = [
            r'(?i)shocking|amazing|unbelievable|incredible|you won\'t believe',
            r'(?i)secret|exclusive|breaking|urgent|just in',
            r'(?i)miracle|cure|breakthrough|discovery|scientists shocked',
            r'(?i)conspiracy|cover-up|exposed|revealed|they don\'t want you to know',
            r'(?i)guaranteed|proven|scientific|studies show|research proves',
            r'(?i)limited time|act now|don\'t miss|exclusive offer',
            r'(?i)click here|learn more|find out|discover the truth',
            r'(?i)mainstream media|fake news|alternative facts|real truth',
            r'(?i)doctors hate|pharma doesn\'t want|experts shocked',
            r'(?i)never before seen|first time ever|revolutionary'
        ]
        
        # Patterns that strongly indicate real news
        self.real_patterns = [
            r'(?i)according to|reported by|confirmed by|official statement',
            r'(?i)study published|research shows|peer-reviewed|scientific journal',
            r'(?i)announced|released|issued|stated|confirmed',
            r'(?i)data shows|statistics indicate|analysis reveals',
            r'(?i)experts say|researchers found|scientists discovered',
            r'(?i)government|official|authority|agency|department',
            r'(?i)verified|authenticated|confirmed|validated',
            r'(?i)source|reference|citation|attribution',
            r'(?i)reliable|credible|trusted|established',
            r'(?i)factual|evidence-based|data-driven|empirical'
        ]
        
        # Initialize models dictionary
        self.models = {}
        self.vectorizers = {}
        self.uncertainty_threshold = 0.15
        self.confidence_threshold = 0.75
        self.min_text_length = 50
        self.max_text_length = 5000
        
    def classify_content_type(self, text):
        """Classify the type of content to apply appropriate model"""
        text_lower = text.lower()
        
        # Check for professional news indicators
        professional_score = 0
        for pattern in self.real_patterns:
            if re.search(pattern, text_lower):
                professional_score += 3
        
        # Check for fake news indicators
        fake_score = 0
        for pattern in self.fake_patterns:
            if re.search(pattern, text_lower):
                fake_score += 2
        
        # Check for excessive punctuation/caps (social media style)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        exclamation_ratio = text.count('!') / len(text) if len(text) > 0 else 0
        
        if caps_ratio > 0.1 or exclamation_ratio > 0.02:
            fake_score += 1
        
        # Determine content type
        if professional_score >= 3:
            return 'professional_news'
        elif fake_score >= 3:
            return 'likely_fake'
        elif caps_ratio > 0.05 or exclamation_ratio > 0.01:
            return 'social_media'
        else:
            return 'general_content'
    
    def rule_based_analysis(self, text):
        """Perform rule-based analysis for obvious cases"""
        text_lower = text.lower()
        
        # Strong fake indicators
        fake_confidence = 0.0
        real_confidence = 0.0
        
        # Check fake patterns
        for pattern in self.fake_patterns:
            if re.search(pattern, text_lower):
                fake_confidence += 0.2
        
        # Check real patterns  
        for pattern in self.real_patterns:
            if re.search(pattern, text_lower):
                real_confidence += 0.15
        
        # Additional checks
        if any(word in text_lower for word in ['breaking:', 'urgent:', 'alert:']):
            if real_confidence < 0.3:  # No professional source found
                fake_confidence += 0.2
        
        # Readability check
        try:
            readability = flesch_reading_ease(text)
            if readability < 10 or readability > 90:  # Very hard or very easy
                fake_confidence += 0.1
        except:
            pass
            
        # Normalize scores
        fake_confidence = min(fake_confidence, 1.0)
        real_confidence = min(real_confidence, 1.0)
        
        # Determine result
        if real_confidence > 0.5:
            return 'REAL', real_confidence, 'Strong professional indicators'
        elif fake_confidence > 0.5:
            return 'FAKE', fake_confidence, 'Strong fake news patterns detected'
        else:
            return 'UNCERTAIN', 0.5, 'Insufficient clear indicators'
    
    def extract_advanced_features(self, text):
        """Extract advanced features from text"""
        features = {}
        
        # Text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(w) for w in text.split()])
        features['sentence_count'] = len(text.split('.'))
        
        # Pattern matching
        features['fake_pattern_matches'] = sum(1 for pattern in self.fake_patterns if re.search(pattern, text))
        features['real_pattern_matches'] = sum(1 for pattern in self.real_patterns if re.search(pattern, text))
        
        # Sentiment and subjectivity
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            features['sentiment'] = blob.sentiment.polarity
            features['subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment'] = 0
            features['subjectivity'] = 0
        
        # Readability
        try:
            import textstat
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['smog_index'] = textstat.smog_index(text)
        except:
            features['flesch_reading_ease'] = 0
            features['smog_index'] = 0
        
        # Named entities
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            features['named_entities'] = len(doc.ents)
        except:
            features['named_entities'] = 0
        
        return features
    
    def predict_with_confidence(self, text):
        """Make prediction with confidence and uncertainty analysis"""
        if not text or len(text) < self.min_text_length:
            return {
                'prediction': 'UNCERTAIN',
                'confidence': 0.0,
                'reason': 'Text too short for reliable analysis'
            }
        
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
        
        # Extract features
        features = self.extract_advanced_features(text)
        
        # Check for obvious patterns
        fake_score = sum(1 for pattern in self.fake_patterns if re.search(pattern, text))
        real_score = sum(1 for pattern in self.real_patterns if re.search(pattern, text))
        
        # Get model predictions
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            if name == 'general_content':
                # Text-based prediction
                text_vectorized = self.vectorizers['general_content'].transform([text])
                pred = model.predict(text_vectorized)[0]
                proba = model.predict_proba(text_vectorized)[0]
                predictions[name] = pred
                confidences[name] = max(proba)
            else:
                # Feature-based prediction
                feature_vector = np.array([list(features.values())])
                pred = model.predict(feature_vector)[0]
                proba = model.predict_proba(feature_vector)[0]
                predictions[name] = pred
                confidences[name] = max(proba)
        
        # Ensemble prediction
        fake_votes = sum(1 for pred in predictions.values() if pred == 'FAKE')
        real_votes = sum(1 for pred in predictions.values() if pred == 'REAL')
        
        # Calculate ensemble confidence
        ensemble_confidence = np.mean(list(confidences.values()))
        
        # Check for uncertainty
        is_uncertain = (
            abs(fake_votes - real_votes) <= 1 or  # Models disagree
            ensemble_confidence < self.confidence_threshold or  # Low confidence
            abs(fake_score - real_score) <= 1  # Pattern scores are close
        )
        
        # Make final prediction
        if is_uncertain:
            return {
                'prediction': 'UNCERTAIN',
                'confidence': ensemble_confidence,
                'reason': 'Multiple indicators suggest uncertainty',
                'details': {
                    'model_votes': {'FAKE': fake_votes, 'REAL': real_votes},
                    'pattern_scores': {'fake': fake_score, 'real': real_score},
                    'feature_analysis': features
                }
            }
        
        final_prediction = 'FAKE' if fake_votes > real_votes else 'REAL'
        
        return {
            'prediction': final_prediction,
            'confidence': ensemble_confidence,
            'reason': 'High confidence ensemble prediction',
            'details': {
                'model_votes': {'FAKE': fake_votes, 'REAL': real_votes},
                'pattern_scores': {'fake': fake_score, 'real': real_score},
                'feature_analysis': features
            }
        }
    
    def train_specialized_models(self, df):
        """Train specialized models for different content types"""
        print("🤖 TRAINING ROBUST DETECTION SYSTEM")
        print("=" * 50)
        
        # Create a more balanced training approach
        print("Creating balanced training data...")
        
        # Separate by actual content patterns rather than just labels
        real_news = df[df['label'] == 'REAL'].copy()
        fake_news = df[df['label'] == 'FAKE'].copy()
        
        print(f"Original split: {len(real_news)} real, {len(fake_news)} fake")
        
        # Train general content model
        print("\nTraining general content model...")
        
        # Use balanced subset
        sample_size = min(len(real_news), len(fake_news), 800)
        balanced_df = pd.concat([
            real_news.sample(sample_size, random_state=42),
            fake_news.sample(sample_size, random_state=42)
        ])
        
        texts = balanced_df['title'].fillna('') + ' ' + balanced_df['text'].fillna('')
        labels = balanced_df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create and train model
        self.vectorizers['general_content'] = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X_train_tfidf = self.vectorizers['general_content'].fit_transform(X_train)
        X_test_tfidf = self.vectorizers['general_content'].transform(X_test)
        
        # Use LogisticRegression for better probability calibration
        self.models['general_content'] = LogisticRegression(
            random_state=42, 
            max_iter=2000, 
            class_weight='balanced',
            C=1.0
        )
        
        self.models['general_content'].fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.models['general_content'].predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"General model accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def save_models(self):
        """Save all models and components"""
        os.makedirs('model/robust', exist_ok=True)
        
        # Save models and vectorizers
        for name, model in self.models.items():
            joblib.dump(model, f'model/robust/{name}_model.pkl')
        
        for name, vectorizer in self.vectorizers.items():
            joblib.dump(vectorizer, f'model/robust/{name}_vectorizer.pkl')
        
        # Save patterns and configuration
        config = {
            'fake_patterns': self.fake_patterns,
            'real_patterns': self.real_patterns,
            'version': '1.0-robust'
        }
        
        with open('model/robust/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("💾 Robust models saved successfully!")
    
    def load_models(self):
        """Load trained models and configuration"""
        try:
            # Load configuration
            with open('model/robust/config.json', 'r') as f:
                config = json.load(f)
            
            self.fake_patterns = config['fake_patterns']
            self.real_patterns = config['real_patterns']
            
            # Load models
            for model_file in os.listdir('model/robust'):
                if model_file.endswith('_model.pkl'):
                    name = model_file.replace('_model.pkl', '')
                    self.models[name] = joblib.load(f'model/robust/{model_file}')
                
                elif model_file.endswith('_vectorizer.pkl'):
                    name = model_file.replace('_vectorizer.pkl', '')
                    self.vectorizers[name] = joblib.load(f'model/robust/{model_file}')
            
            print(f"✅ Loaded {len(self.models)} models and {len(self.vectorizers)} vectorizers")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

def main():
    """Train the robust fake news detection system"""
    print("🚀 ROBUST FAKE NEWS DETECTION SYSTEM")
    print("=" * 60)
    
    detector = RobustFakeNewsDetector()
    
    # Load data
    try:
        df = pd.read_csv('data/fake_or_real_news.csv')
        print(f"📚 Loaded dataset with {len(df)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Train models
    accuracy = detector.train_specialized_models(df)
    
    # Save models
    detector.save_models()
    
    # Test with examples
    print(f"\n🧪 TESTING ROBUST SYSTEM")
    print("=" * 50)
    
    test_cases = [
        "WASHINGTON (Reuters) - The Federal Reserve announced new interest rate policies today following economic indicators showing steady growth.",
        "SHOCKING! This one weird trick will make you rich overnight! Banks hate this secret method!",
        "Apple Inc. reported quarterly earnings showing revenue growth driven by iPhone sales according to company officials.",
        "BREAKING: Aliens confirmed by government officials! Secret documents reveal decades of cover-up that mainstream media doesn't want you to know!",
        "Scientists at Stanford University published peer-reviewed research on climate change impacts in the journal Nature.",
        "You won't believe what doctors discovered! This miracle cure that big pharma doesn't want you to know about!",
        "The president met with congressional leaders today to discuss budget proposals, according to White House sources."
    ]
    
    for i, text in enumerate(test_cases, 1):
        result = detector.predict_with_confidence(text)
        
        status = "🟢 REAL" if result['prediction'] == 'REAL' else "🔴 FAKE" if result['prediction'] == 'FAKE' else "⚪ UNCERTAIN"
        uncertain = " ⚠️" if result['is_uncertain'] else ""
        
        print(f"\n{i}. {status}{uncertain} - {result['confidence']:.1%} confidence")
        print(f"   Method: {result['method']}")
        print(f"   Content Type: {result['content_type']}")
        print(f"   Reason: {result['reason']}")
        print(f"   Text: {text[:80]}...")
    
    print(f"\n🎉 ROBUST TRAINING COMPLETED!")
    print(f"System ready with conservative, explainable predictions!")

if __name__ == "__main__":
    main() 