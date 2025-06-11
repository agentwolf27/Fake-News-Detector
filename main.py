#!/usr/bin/env python3
"""
Fake News Detector - Main Entry Point
"""

import sys
import os
import argparse

def download_dataset():
    """Download fake news dataset from Kaggle"""
    print("📥 Downloading fake news dataset...")
    os.system("python model/download_dataset.py")

def train_model():
    """Train the fake news detection model"""
    print("🤖 Training fake news detection model...")
    os.system("python model/train_model.py")

def run_app():
    """Run the Flask web application"""
    print("🚀 Starting the web application...")
    os.system("python app/app.py")

def run_tests():
    """Run some test predictions"""
    from utils.preprocess import get_prediction_details
    
    test_cases = [
        {
            "text": "Scientists at Harvard have discovered a new treatment for diabetes that shows promising results in clinical trials.",
            "expected": "REAL"
        },
        {
            "text": "BREAKING: Aliens have landed in Area 51 and are demanding to speak with our planet's manager!",
            "expected": "FAKE"
        },
        {
            "text": "The stock market closed up 2% today following positive economic indicators and strong quarterly earnings reports.",
            "expected": "REAL"
        },
        {
            "text": "Eating this one weird fruit will cure all diseases and make you immortal, doctors hate this trick!",
            "expected": "FAKE"
        }
    ]
    
    print("🧪 Running test predictions...")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        result = get_prediction_details(test["text"])
        status = "✅" if result["prediction"] == test["expected"] else "❌"
        
        print(f"\nTest {i}: {status}")
        print(f"Text: {test['text'][:50]}...")
        print(f"Expected: {test['expected']}")
        print(f"Predicted: {result['prediction']} ({result['confidence']}% confidence)")

def main():
    parser = argparse.ArgumentParser(description="Fake News Detector")
    parser.add_argument('action', 
                       choices=['download', 'train', 'run', 'test', 'all'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'download':
        download_dataset()
    elif args.action == 'train':
        train_model()
    elif args.action == 'run':
        run_app()
    elif args.action == 'test':
        run_tests()
    elif args.action == 'all':
        print("🔄 Running complete pipeline...")
        download_dataset()
        print("\n" + "="*50)
        train_model()
        print("\n" + "="*50)
        run_tests()
        print("\n" + "="*50)
        run_app()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🔍 Fake News Detector")
        print("=" * 30)
        print("Usage:")
        print("  python main.py download - Download Kaggle dataset")
        print("  python main.py train    - Train the model")
        print("  python main.py run      - Run the web app")
        print("  python main.py test     - Test predictions")
        print("  python main.py all      - Do everything")
        print()
        print("Quick start: python main.py all")
    else:
        main() 