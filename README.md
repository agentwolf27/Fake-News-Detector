# 🔍 Fake News Detector

A full-stack web application that uses machine learning to detect fake news articles and headlines. Built with Python, Flask, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Real-time Prediction**: Instantly analyze news articles and headlines
- **Confidence Scoring**: Get probability scores for predictions
- **Beautiful UI**: Modern, responsive web interface with color-coded results
- **REST API**: JSON API endpoint for programmatic access
- **Docker Support**: Easy deployment with containerization
- **High Accuracy**: TF-IDF + Logistic Regression model with balanced training
- **Interactive Examples**: Try pre-loaded examples to test the system

## 🚀 Quick Start

### Option 1: Run Everything (Recommended)
```bash
# Clone and setup
git clone <repository-url>
cd fake-news-detector
pip install -r requirements.txt

# Train model and start web app
python main.py all
```

### Option 2: Step by Step
```bash
# Install dependencies
pip install -r requirements.txt

# Download real dataset from Kaggle (optional)
python main.py download

# Train the model
python model/train_model.py

# Start web application
python app/app.py
```

### Option 3: Docker
```bash
# Build and run with Docker
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector
```

Open your browser to `http://localhost:5000` 🎉

## 📁 Project Structure

```
project/
├── main.py                 # Main entry point
├── app/
│   └── app.py             # Flask web application
├── model/
│   ├── train_model.py     # Model training script
│   ├── fake_news_model.pkl    # Trained model (generated)
│   └── vectorizer.pkl     # TF-IDF vectorizer (generated)
├── utils/
│   └── preprocess.py      # Text preprocessing utilities
├── templates/
│   └── index.html         # Web interface template
├── static/
│   └── style.css          # Styling and responsive design
├── data/
│   └── fake_or_real_news.csv # Training dataset
├── test_script.py         # Testing and benchmarking
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md            # This file
```

## 🤖 How It Works

### 1. **Data Processing**
- Combines news headlines and article text
- Cleans text (removes HTML, URLs, special characters)
- Removes stopwords and normalizes text

### 2. **Feature Extraction**
- Uses TF-IDF (Term Frequency-Inverse Document Frequency)
- Extracts 10,000 most important features
- Includes both unigrams and bigrams

### 3. **Machine Learning**
- Logistic Regression classifier with balanced class weights
- 80/20 train-test split with stratification
- Achieves high accuracy on validation data

### 4. **Prediction Pipeline**
- Real-time text preprocessing
- Feature vectorization
- Probability calculation with confidence scores

## 🎯 Usage Examples

### Web Interface
1. Open `http://localhost:5000`
2. Paste news article or headline
3. Click "Analyze News"
4. View prediction with confidence score

### API Usage
```python
import requests

response = requests.post('http://localhost:5000/api/predict', 
                        json={'text': 'Your news article here'})
result = response.json()
print(f"Prediction: {result['prediction']} ({result['confidence']}%)")
```

### CLI Testing
```bash
# Test local predictions
python test_script.py local

# Interactive mode
python test_script.py interactive

# Performance benchmark
python test_script.py benchmark

# Test API endpoint
python test_script.py api
```

## 📊 Model Performance

Based on the training dataset:

- **Accuracy**: ~85-90% on test set
- **Features**: 10,000 TF-IDF features
- **Algorithm**: Logistic Regression with balanced class weights
- **Preprocessing**: Text cleaning, stopword removal, normalization

### Sample Predictions:

✅ **REAL (92% confidence)**
> "Scientists at MIT develop new water purification technology using solar energy"

❌ **FAKE (88% confidence)**  
> "Government plans to replace all birds with surveillance drones"

## 🛠️ Configuration

### Environment Variables
- `FLASK_ENV`: Set to `development` for debug mode
- `PORT`: Custom port (default: 5000)

### Model Parameters
Edit `model/train_model.py` to adjust:
- `max_features`: Number of TF-IDF features (default: 10,000)
- `ngram_range`: N-gram range (default: (1,2))
- `test_size`: Train-test split ratio (default: 0.2)

## 🐳 Docker Deployment

```bash
# Build image
docker build -t fake-news-detector .

# Run container
docker run -d -p 5000:5000 --name fake-news-app fake-news-detector

# View logs
docker logs fake-news-app

# Stop container
docker stop fake-news-app
```

## 🧪 Testing

Run comprehensive tests:
```bash
# All tests
python test_script.py all

# Specific tests
python test_script.py local      # Local predictions
python test_script.py api        # API endpoint
python test_script.py benchmark  # Performance test
```

## 📈 API Endpoints

### `POST /api/predict`
Predict news authenticity

**Request:**
```json
{
  "text": "Your news article text here"
}
```

**Response:**
```json
{
  "text": "Your news article text here...",
  "prediction": "REAL",
  "confidence": 92.5,
  "confidence_score": 0.925
}
```

### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "message": "Fake News Detector is running"
}
```

## 🚧 Limitations & Disclaimers

- **Training Data**: Limited to the provided dataset
- **Accuracy**: Not 100% accurate - use as guidance only
- **Bias**: May reflect biases present in training data
- **Context**: Cannot understand complex context or satire
- **Languages**: Optimized for English text only

**⚠️ Important**: Always verify news from multiple reliable sources. This tool is for educational and assistance purposes only.

## 🛣️ Future Enhancements

- [ ] **BERT Integration**: Upgrade to transformer-based models
- [ ] **Source Verification**: Check against trusted news sources
- [ ] **Explanation Features**: Highlight suspicious phrases
- [ ] **Multi-language Support**: Support for multiple languages
- [ ] **Real-time Updates**: Continuous learning from new data
- [ ] **Browser Extension**: Chrome/Firefox extension
- [ ] **Mobile App**: React Native or Flutter app

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Fake News Detection datasets from various sources
- **Libraries**: scikit-learn, Flask, pandas, and other open-source projects
- **Inspiration**: The need for reliable news verification tools

## 📞 Support

- **Issues**: Report bugs via GitHub Issues
- **Questions**: Use GitHub Discussions
- **Email**: Contact maintainers for urgent issues

---

**Built with ❤️ for a more informed world** 🌍 