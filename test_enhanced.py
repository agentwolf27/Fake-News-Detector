#!/usr/bin/env python3
"""
Enhanced Fake News Detector Test Script
Tests the ensemble prediction system with uncertainty indicators
"""

from utils.preprocess import predict_ensemble
import sys

def test_enhanced_system():
    """Test the enhanced fake news detection system"""
    
    print("🚀 ENHANCED FAKE NEWS DETECTOR COMPREHENSIVE TEST")
    print("=" * 70)
    
    test_cases = [
        {
            'text': 'WASHINGTON (Reuters) - The Federal Reserve announced new interest rate policies today following economic indicators.',
            'expected': 'Should be REAL (professional format)'
        },
        {
            'text': 'Apple Inc. reported strong quarterly earnings with revenue growth of 15% driven by iPhone sales.',
            'expected': 'May be uncertain (good content, no source)'
        },
        {
            'text': 'BREAKING: Aliens discovered living in pizza restaurants worldwide! Government coverup exposed!',
            'expected': 'Should be FAKE (conspiracy content)'
        },
        {
            'text': 'Miracle diet pill melts fat overnight - doctors hate this simple trick that pharma companies hide!',
            'expected': 'Should be FAKE (clickbait patterns)'
        },
        {
            'text': 'The earth is round and orbits the sun according to scientific consensus.',
            'expected': 'May be uncertain (simple factual statement)'
        },
        {
            'text': 'Scientists at Stanford University published peer-reviewed research on climate change impacts in Nature journal.',
            'expected': 'Should be REAL (academic source, proper citation)'
        },
        {
            'text': 'SHOCKING! This one weird trick will make you rich in 30 days! Banks hate it!',
            'expected': 'Should be FAKE (obvious scam pattern)'
        }
    ]
    
    print(f"Testing {len(test_cases)} cases with ensemble models:\n")
    
    for i, case in enumerate(test_cases, 1):
        text = case['text']
        expected = case['expected']
        
        print(f"Test {i}: {expected}")
        print(f"Text: \"{text[:80]}...\"" if len(text) > 80 else f"Text: \"{text}\"")
        
        # Get enhanced prediction
        result = predict_ensemble(text)
        
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}\n")
            continue
        
        # Display results
        prediction = result['prediction']
        confidence = result['confidence']
        confidence_level = result['confidence_level']
        is_uncertain = result['is_uncertain']
        models_agree = result['models_agree']
        
        # Format output
        status_icon = '🟢' if prediction == 'REAL' else '🔴'
        uncertainty_icon = ' ⚠️' if is_uncertain else ''
        agreement_icon = '🤝' if models_agree else '⚖️'
        
        print(f"📊 RESULT: {status_icon} {prediction}{uncertainty_icon}")
        print(f"   Confidence: {confidence}% ({confidence_level})")
        print(f"   Model Agreement: {agreement_icon} {'Yes' if models_agree else 'No'}")
        
        # Model breakdown
        print("   📰 Format Model:", 
              f"{result['format_prediction']} ({result['format_fake_prob']}% Fake, {result['format_real_prob']}% Real)")
        
        if result.get('content_prediction'):
            print("   📝 Content Model:", 
                  f"{result['content_prediction']} ({result['content_fake_prob']}% Fake, {result['content_real_prob']}% Real)")
        
        print("   🎯 Ensemble:", 
              f"{result['ensemble_fake_prob']}% Fake, {result['ensemble_real_prob']}% Real")
        
        # Analysis
        if is_uncertain:
            print("   ⚠️  HIGH UNCERTAINTY - Consider manual review")
        if not models_agree:
            print("   ⚖️  MODELS DISAGREE - Mixed signals detected")
        
        print()
    
    print("=" * 70)
    print("✅ Enhanced testing complete!")
    print("\n🔍 Key Features Demonstrated:")
    print("   ✅ Dual-model ensemble predictions")
    print("   ✅ Uncertainty detection and alerts")
    print("   ✅ Model agreement analysis")
    print("   ✅ Confidence level categorization")
    print("   ✅ Format vs content analysis breakdown")
    print("   ✅ JSON-serializable results for API")

def test_api_compatibility():
    """Test API response format"""
    print("\n🔌 API COMPATIBILITY TEST")
    print("-" * 40)
    
    import requests
    import json
    
    try:
        # Test API endpoint
        response = requests.post('http://localhost:8080/api/predict', 
                               json={'text': 'Test news article content'})
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API endpoint working")
            print("✅ JSON serialization working")
            print(f"✅ Response contains: {', '.join(data.keys())}")
        else:
            print(f"❌ API error: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️  API test failed: {e}")
        print("   (Make sure the Flask app is running)")

if __name__ == "__main__":
    test_enhanced_system()
    test_api_compatibility() 