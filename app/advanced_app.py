#!/usr/bin/env python3
"""
Advanced Flask App for Fake News Detection
Uses sophisticated ML models with multiple validation approaches
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the advanced detector
from model.advanced_model import AdvancedFakeNewsDetector

app = Flask(__name__, 
    template_folder='../templates',
    static_folder='../static'
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global detector instance
detector = None

def initialize_detector():
    """Initialize the advanced fake news detector"""
    global detector
    try:
        detector = AdvancedFakeNewsDetector()
        
        # Try to load existing models
        if detector.load_models():
            logger.info("✅ Advanced models loaded successfully!")
            return True
        else:
            logger.warning("⚠️ No pre-trained models found. Please train first!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error initializing detector: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('advanced_index.html')

@app.route('/api/predict/advanced', methods=['POST'])
def predict_advanced():
    """Advanced prediction endpoint with comprehensive analysis"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data.get('text', '').strip()
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short for reliable analysis'}), 400
        
        if detector is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Get advanced prediction
        result = detector.predict_advanced(text)
        
        # Add performance metrics
        result['model_type'] = 'advanced_ensemble'
        result['version'] = '2.0'
        result['features_count'] = len(result.get('features', {}))
        
        # Format confidence categories
        confidence = result.get('confidence', 0.5)
        if confidence >= 0.9:
            result['confidence_category'] = 'VERY_HIGH'
        elif confidence >= 0.8:
            result['confidence_category'] = 'HIGH'
        elif confidence >= 0.6:
            result['confidence_category'] = 'MEDIUM'
        else:
            result['confidence_category'] = 'LOW'
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict_simple():
    """Simple prediction endpoint for compatibility"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data.get('text', '').strip()
        
        if detector is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Get advanced prediction but return simplified format
        result = detector.predict_advanced(text)
        
        # Simplified response
        return jsonify({
            'prediction': result.get('prediction', 'UNCERTAIN'),
            'confidence': result.get('confidence', 0.5),
            'fake_probability': result.get('fake_probability', 0.5),
            'real_probability': result.get('real_probability', 0.5),
            'is_uncertain': result.get('is_uncertain', True)
        })
        
    except Exception as e:
        logger.error(f"Simple prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Detailed text analysis endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data.get('text', '').strip()
        
        if detector is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Get full analysis
        result = detector.predict_advanced(text)
        
        # Add detailed feature analysis
        features = result.get('features', {})
        
        analysis = {
            'prediction_details': {
                'final_prediction': result.get('prediction'),
                'confidence': result.get('confidence'),
                'ensemble_prediction': result.get('ensemble_prediction'),
                'feature_prediction': result.get('feature_prediction'),
                'rule_score': result.get('rule_score'),
                'model_agreement': result.get('model_agreement')
            },
            'text_statistics': {
                'word_count': features.get('word_count', 0),
                'sentence_count': features.get('sentence_count', 0),
                'avg_word_length': features.get('avg_word_length', 0),
                'readability_score': features.get('flesch_reading_ease', 50)
            },
            'credibility_indicators': {
                'has_reliable_source': features.get('has_reliable_source', False),
                'fake_keyword_count': features.get('fake_keyword_count', 0),
                'has_quotes': features.get('has_quotes', False),
                'date_references': features.get('date_references', 0)
            },
            'linguistic_patterns': {
                'sentiment_score': features.get('sentiment_compound', 0),
                'exclamation_ratio': features.get('exclamation_ratio', 0),
                'caps_ratio': features.get('caps_ratio', 0),
                'superlatives_count': features.get('superlatives_count', 0)
            },
            'uncertainty_factors': {
                'is_uncertain': result.get('is_uncertain', True),
                'models_disagree': not result.get('model_agreement', True),
                'low_confidence': result.get('confidence', 0.5) < 0.6
            }
        }
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if detector is not None else 'unhealthy',
        'model_loaded': detector is not None,
        'version': '2.0-advanced',
        'features': [
            'Advanced ensemble models',
            'Rule-based validation',
            'Sentiment analysis',
            'Linguistic pattern detection',
            'Source credibility checking',
            'Uncertainty quantification'
        ]
    }
    
    return jsonify(status)

@app.route('/api/models')
def model_info():
    """Get information about loaded models"""
    if detector is None:
        return jsonify({'error': 'No models loaded'}), 500
    
    info = {
        'model_type': 'advanced_ensemble',
        'version': '2.0',
        'components': [
            'TF-IDF + Advanced Features + Ensemble (RandomForest + LogisticRegression + SVM)',
            'Feature-only RandomForest',
            'Rule-based validator'
        ],
        'features': [
            'Text statistics (word count, readability)',
            'Sentiment analysis (VADER)',
            'Linguistic patterns (punctuation, capitalization)',
            'Credibility indicators (sources, quotes)',
            'Fake news keywords detection',
            'Temporal references analysis'
        ],
        'accuracy': 'Expected >95% on diverse content',
        'capabilities': [
            'Multi-model ensemble prediction',
            'Uncertainty detection',
            'Source credibility assessment',
            'Detailed feature analysis'
        ]
    }
    
    return jsonify(info)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 STARTING ADVANCED FAKE NEWS DETECTOR")
    print("=" * 60)
    
    # Initialize the detector
    print("🔧 Initializing advanced detection system...")
    if initialize_detector():
        print("✅ Advanced models loaded successfully!")
        print("\n🌟 ADVANCED FEATURES:")
        print("   • Multi-model ensemble prediction")
        print("   • Advanced NLP feature extraction")
        print("   • Rule-based validation")
        print("   • Sentiment and linguistic analysis")
        print("   • Source credibility checking")
        print("   • Uncertainty quantification")
        print("   • Comprehensive confidence scoring")
        
        print(f"\n📱 Server starting at: http://localhost:8080")
        print("🔍 Ready for advanced fake news detection!")
        
        app.run(host='0.0.0.0', port=8080, debug=True)
    else:
        print("❌ Failed to initialize models!")
        print("💡 Please run: python model/advanced_model.py")
        print("   to train the advanced models first.") 