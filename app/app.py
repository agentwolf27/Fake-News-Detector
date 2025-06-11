from flask import Flask, render_template, request, jsonify
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocess import get_prediction_details, predict_ensemble

app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    """Handle prediction requests from the web interface"""
    try:
        text = request.form['text']
        
        if not text.strip():
            return render_template('index.html', 
                                 error="Please enter some text to analyze")
        
        # Get enhanced prediction with ensemble approach
        result = predict_ensemble(text)
        
        if 'error' in result:
            return render_template('index.html', error=result['error'])
        
        # Format results for display
        prediction = result['prediction']
        confidence = result['confidence']
        confidence_level = result['confidence_level']
        is_uncertain = result['is_uncertain']
        models_agree = result['models_agree']
        
        # Determine display style based on prediction and confidence
        if prediction == 'FAKE':
            status_class = 'fake'
            status_icon = '🔴'
        else:
            status_class = 'real'
            status_icon = '🟢'
        
        # Add uncertainty indicator
        if is_uncertain:
            status_icon += ' ⚠️'
            uncertainty_message = "This prediction has high uncertainty. Consider the content carefully."
        else:
            uncertainty_message = None
        
        # Model agreement indicator
        agreement_message = None
        if result.get('content_prediction') is not None:
            if models_agree:
                agreement_message = "Both format and content models agree on this prediction."
            else:
                agreement_message = f"Models disagree: Format={result['format_prediction']}, Content={result['content_prediction']}"
        
        return render_template('index.html',
                             prediction=prediction,
                             confidence=confidence,
                             confidence_level=confidence_level,
                             status_class=status_class,
                             status_icon=status_icon,
                             text=text,
                             is_uncertain=is_uncertain,
                             uncertainty_message=uncertainty_message,
                             models_agree=models_agree,
                             agreement_message=agreement_message,
                             ensemble_fake_prob=result.get('ensemble_fake_prob'),
                             ensemble_real_prob=result.get('ensemble_real_prob'),
                             format_fake_prob=result.get('format_fake_prob'),
                             format_real_prob=result.get('format_real_prob'),
                             content_fake_prob=result.get('content_fake_prob'),
                             content_real_prob=result.get('content_real_prob'))
        
    except Exception as e:
        return render_template('index.html', 
                             error=f"An error occurred: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Get enhanced prediction
        result = predict_ensemble(text)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Return enhanced API response
        api_response = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'confidence_level': result['confidence_level'],
            'is_uncertain': result['is_uncertain'],
            'models_agree': result['models_agree'],
            'text': result['text'],
            'ensemble_probabilities': {
                'fake': result['ensemble_fake_prob'],
                'real': result['ensemble_real_prob']
            },
            'model_breakdown': {
                'format_model': {
                    'prediction': result['format_prediction'],
                    'fake_prob': result['format_fake_prob'],
                    'real_prob': result['format_real_prob']
                }
            }
        }
        
        # Add content model results if available
        if result.get('content_prediction') is not None:
            api_response['model_breakdown']['content_model'] = {
                'prediction': result['content_prediction'],
                'fake_prob': result['content_fake_prob'],
                'real_prob': result['content_real_prob']
            }
        
        return jsonify(api_response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/simple', methods=['POST'])
def api_predict_simple():
    """Simple API endpoint for basic predictions (backward compatibility)"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        result = get_prediction_details(text)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Return simple response for compatibility
        return jsonify({
            'text': result['text'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'confidence_score': result['confidence'] / 100
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'fake-news-detector',
        'models': {
            'format_model': 'available',
            'content_model': 'available' if os.path.exists('model/content_model.pkl') else 'not_available'
        }
    })

@app.route('/api/models')
def api_models():
    """Get information about available models"""
    models_info = {
        'format_model': {
            'description': 'Focuses on source attribution and professional formatting',
            'available': os.path.exists('model/fake_news_model.pkl')
        },
        'content_model': {
            'description': 'Focuses on content quality and writing patterns',
            'available': os.path.exists('model/content_model.pkl')
        },
        'ensemble': {
            'description': 'Combines both models for enhanced accuracy',
            'available': True
        }
    }
    
    return jsonify(models_info)

@app.errorhandler(404)
def not_found(error):
    """
    404 error handler
    """
    return render_template('index.html', 
                         result={'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """
    500 error handler
    """
    return render_template('index.html', 
                         result={'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check if model files exist
    model_path = os.path.join('model', 'fake_news_model.pkl')
    vectorizer_path = os.path.join('model', 'vectorizer.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("❌ Model files not found!")
        print("Please run the following command first:")
        print("python model/train_model.py")
        print()
        print("This will train the model and create the required .pkl files.")
        sys.exit(1)
    
    print("🚀 Starting Enhanced Fake News Detector...")
    print("📱 Open your browser and go to: http://localhost:8080")
    print("🔍 Ready to detect fake news with ensemble models!")
    print("✨ Features:")
    print("   - Ensemble predictions (format + content)")
    print("   - Uncertainty indicators")
    print("   - Model agreement analysis")
    print("   - Enhanced confidence levels")
    
    app.run(debug=True, host='0.0.0.0', port=8080) 