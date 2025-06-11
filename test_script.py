#!/usr/bin/env python3
"""
Test Script for Fake News Detector
"""

import sys
import requests
import json
from utils.preprocess import get_prediction_details

def test_local_prediction():
    """Test local prediction without Flask app"""
    print("🧪 Testing Local Predictions")
    print("=" * 40)
    
    test_cases = [
        "Breaking: Scientists discover cure for aging using simple household ingredient",
        "Federal Reserve announces new interest rate policy following economic indicators",
        "Miracle diet pill melts fat overnight - doctors hate this one trick!",
        "University researchers publish peer-reviewed study on climate change impacts",
        "Aliens spotted in downtown New York City demanding pizza",
        "Tech company reports quarterly earnings exceeding analyst expectations"
    ]
    
    for i, text in enumerate(test_cases, 1):
        result = get_prediction_details(text)
        
        color = "🔴" if result['prediction'] == 'FAKE' else "🟢"
        print(f"\n{i}. {color} {result['prediction']} ({result['confidence']}%)")
        print(f"   Text: {text[:60]}...")

def test_api_endpoint():
    """Test Flask API endpoint"""
    print("\n🌐 Testing API Endpoint")
    print("=" * 40)
    
    # Test data
    test_data = {
        "text": "Scientists at MIT develop revolutionary water purification technology using solar energy"
    }
    
    try:
        response = requests.post(
            "http://localhost:8080/api/predict",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Response: {result['prediction']} ({result['confidence']}%)")
        else:
            print(f"❌ API Error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure Flask app is running on localhost:8080")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def interactive_mode():
    """Interactive CLI mode for testing"""
    print("\n💬 Interactive Mode")
    print("=" * 40)
    print("Enter news text to analyze (or 'quit' to exit):")
    
    while True:
        try:
            text = input("\n📰 News text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
                
            if not text:
                print("Please enter some text.")
                continue
                
            result = get_prediction_details(text)
            
            # Format output with colors
            color = "🔴" if result['prediction'] == 'FAKE' else "🟢"
            confidence_bar = "█" * int(result['confidence'] / 5)  # Visual confidence bar
            
            print(f"\n{color} Prediction: {result['prediction']}")
            print(f"📊 Confidence: {result['confidence']}% {confidence_bar}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def benchmark_performance():
    """Benchmark prediction performance"""
    import time
    
    print("\n⚡ Performance Benchmark")
    print("=" * 40)
    
    test_text = "This is a sample news article for performance testing. " * 10
    num_predictions = 50
    
    start_time = time.time()
    
    for i in range(num_predictions):
        get_prediction_details(test_text)
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_predictions} predictions")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_predictions
    
    print(f"\n📈 Results:")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Average time per prediction: {avg_time:.3f} seconds")
    print(f"   Predictions per second: {1/avg_time:.1f}")

def main():
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        
        if action == 'local':
            test_local_prediction()
        elif action == 'api':
            test_api_endpoint()
        elif action == 'interactive':
            interactive_mode()
        elif action == 'benchmark':
            benchmark_performance()
        elif action == 'all':
            test_local_prediction()
            test_api_endpoint()
            benchmark_performance()
        else:
            print(f"Unknown action: {action}")
            show_usage()
    else:
        show_usage()

def show_usage():
    print("🧪 Fake News Detector Test Script")
    print("=" * 35)
    print("Usage:")
    print("  python test_script.py local       - Test local predictions")
    print("  python test_script.py api         - Test API endpoint")
    print("  python test_script.py interactive - Interactive mode")
    print("  python test_script.py benchmark   - Performance benchmark")
    print("  python test_script.py all         - Run all tests")
    print()
    print("Quick start: python test_script.py local")

if __name__ == "__main__":
    main() 