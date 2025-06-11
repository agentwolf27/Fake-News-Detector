from flask import Flask, request, jsonify
from utils import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    result = predict(data['text'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 