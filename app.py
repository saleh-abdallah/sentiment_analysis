# API Development and Containerization
# Develop a Flask API that can receive a tweet, process it, and return the sentiment classification.

from flask import Flask, request, jsonify
from flask_cors import CORS  
import joblib

app = Flask(__name__)
CORS(app)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return 'Sentiment API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet = data['text']
    vec = vectorizer.transform([tweet])
    pred = model.predict(vec)[0]
    sentiment = ['Negative', 'Positive'][pred]
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


