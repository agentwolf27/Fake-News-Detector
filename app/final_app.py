#!/usr/bin/env python3
"""
Final Production-Ready Fake News Detector
Uses the robust detection system with conservative, explainable predictions
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the robust detector
from model.robust_detector import RobustFakeNewsDetector

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
    """Initialize the robust fake news detector"""
    global detector
    try:
        detector = RobustFakeNewsDetector()
        
        # Try to load existing models
        if detector.load_models():
            logger.info("✅ Robust models loaded successfully!")
            return True
        else:
            logger.warning("⚠️ No pre-trained models found. Please train first!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error initializing detector: {e}")
        return False

@app.route('/')
def index():
    """Main page with robust detection interface"""
    return render_template('final_index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint using robust analysis"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data.get('text', '').strip()
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short for reliable analysis (minimum 10 characters)'}), 400
        
        if detector is None:
            return jsonify({'error': 'Detection system not initialized'}), 500
        
        # Get robust prediction
        result = detector.predict_with_confidence(text)
        
        # Format response for frontend
        response = {
            'prediction': result.get('prediction', 'UNCERTAIN'),
            'confidence': result.get('confidence', 0.5),
            'is_uncertain': result.get('is_uncertain', True),
            'method': result.get('method', 'unknown'),
            'reason': result.get('reason', 'No reason provided'),
            'content_type': result.get('content_type', 'unknown'),
            'rule_prediction': result.get('rule_prediction', 'UNCERTAIN'),
            'rule_confidence': result.get('rule_confidence', 0.5),
            'system': 'robust_v1.0'
        }
        
        # Add probability breakdown
        if result['prediction'] == 'REAL':
            response['real_probability'] = result['confidence']
            response['fake_probability'] = 1 - result['confidence']
        elif result['prediction'] == 'FAKE':
            response['fake_probability'] = result['confidence']
            response['real_probability'] = 1 - result['confidence']
        else:  # UNCERTAIN
            response['real_probability'] = 0.5
            response['fake_probability'] = 0.5
        
        # Add features for analysis
        features = result.get('features', {})
        response['analysis'] = {
            'word_count': features.get('word_count', 0),
            'has_professional_source': features.get('has_professional_source', False),
            'fake_keywords': features.get('fake_keywords', 0),
            'has_attribution': features.get('has_attribution', False),
            'sentiment_score': features.get('sentiment_compound', 0),
            'caps_ratio': features.get('caps_ratio', 0)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'prediction': 'ERROR',
            'confidence': 0.0,
            'error': f'Analysis failed: {str(e)}',
            'is_uncertain': True
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Detailed analysis endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data.get('text', '').strip()
        
        if detector is None:
            return jsonify({'error': 'Detection system not initialized'}), 500
        
        # Get full analysis
        result = detector.predict_with_confidence(text)
        
        # Enhanced analysis response
        analysis = {
            'prediction_summary': {
                'final_verdict': result.get('prediction', 'UNCERTAIN'),
                'confidence_level': result.get('confidence', 0.5),
                'is_uncertain': result.get('is_uncertain', True),
                'analysis_method': result.get('method', 'unknown')
            },
            'content_classification': {
                'content_type': result.get('content_type', 'unknown'),
                'detected_patterns': result.get('reason', 'No patterns detected')
            },
            'model_breakdown': {
                'rule_based': {
                    'prediction': result.get('rule_prediction', 'UNCERTAIN'),
                    'confidence': result.get('rule_confidence', 0.5)
                }
            },
            'feature_analysis': result.get('features', {}),
            'explanation': {
                'why_this_verdict': result.get('reason', 'Analysis incomplete'),
                'key_indicators': [],
                'confidence_factors': []
            }
        }
        
        # Add explanatory details
        features = result.get('features', {})
        
        # Key indicators
        if features.get('has_professional_source', False):
            analysis['explanation']['key_indicators'].append('✅ Contains professional news source')
        if features.get('fake_keywords', 0) > 0:
            analysis['explanation']['key_indicators'].append(f'⚠️ Contains {features["fake_keywords"]} fake news keywords')
        if features.get('has_attribution', False):
            analysis['explanation']['key_indicators'].append('✅ Contains proper attribution')
        
        # Confidence factors
        if result.get('is_uncertain', True):
            analysis['explanation']['confidence_factors'].append('⚠️ Analysis shows uncertainty')
        if result.get('confidence', 0.5) > 0.8:
            analysis['explanation']['confidence_factors'].append('✅ High confidence prediction')
        elif result.get('confidence', 0.5) < 0.6:
            analysis['explanation']['confidence_factors'].append('⚠️ Low confidence - treat with caution')
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """System health check"""
    status = {
        'status': 'healthy' if detector is not None else 'unhealthy',
        'system': 'robust_fake_news_detector',
        'version': '1.0',
        'model_loaded': detector is not None,
        'features': [
            'Rule-based pattern detection',
            'Content type classification',
            'Conservative uncertainty handling',
            'Explainable predictions',
            'Balanced training approach'
        ],
        'accuracy': 'Conservative, bias-resistant approach'
    }
    
    return jsonify(status)

@app.route('/api/test', methods=['GET'])
def test_examples():
    """Get test examples for demonstration"""
    examples = [
        {
            'category': 'Real News',
            'text': 'WASHINGTON (Reuters) - The Federal Reserve announced new interest rate policies today following economic indicators showing steady growth.',
            'expected': 'REAL',
            'description': 'Professional news with proper attribution'
        },
        {
            'category': 'Obvious Fake',
            'text': 'SHOCKING! This one weird trick will make you rich overnight! Banks hate this secret method that governments don\'t want you to know!',
            'expected': 'FAKE',
            'description': 'Classic clickbait with multiple fake news patterns'
        },
        {
            'category': 'Medical Scam',
            'text': 'You won\'t believe what doctors discovered! This miracle cure that big pharma doesn\'t want you to know about will cure all diseases naturally!',
            'expected': 'FAKE',
            'description': 'Medical misinformation with conspiracy elements'
        },
        {
            'category': 'Neutral/Uncertain',
            'text': 'Apple Inc. reported quarterly earnings showing revenue growth driven by iPhone sales according to company officials.',
            'expected': 'UNCERTAIN',
            'description': 'Corporate news without clear source attribution'
        },
        {
            'category': 'Conspiracy Theory',
            'text': 'BREAKING: Government officials confirm alien contact! Secret documents reveal decades of cover-up that mainstream media refuses to report!',
            'expected': 'FAKE',
            'description': 'Conspiracy theory with classic fake news markers'
        }
    ]
    
    return jsonify({'examples': examples})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 FINAL FAKE NEWS DETECTOR v1.0")
    print("=" * 60)
    print("🔍 Robust • Conservative • Explainable")
    print()
    
    # Initialize the detector
    print("🔧 Initializing robust detection system...")
    if initialize_detector():
        print("✅ System ready!")
        print()
        print("🌟 KEY FEATURES:")
        print("   • Rule-based pattern detection")
        print("   • Content type classification") 
        print("   • Conservative uncertainty handling")
        print("   • Explainable decision making")
        print("   • Bias-resistant training approach")
        print("   • Production-ready reliability")
        print()
        print(f"📱 Access at: http://localhost:8080")
        print("🎯 Ready for accurate fake news detection!")
        print()
        
        app.run(host='0.0.0.0', port=8080, debug=True)
    else:
        print("❌ Failed to initialize detection system!")
        print()
        print("💡 TO FIX THIS:")
        print("   1. Run: python model/robust_detector.py")
        print("   2. Wait for training to complete")
        print("   3. Restart this application")
        print()
        sys.exit(1) 