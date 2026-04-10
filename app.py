import os
from flask import Flask, request, jsonify
from transformers import pipeline
from waitress import serve
from dotenv import load_dotenv

# LAB 2 STEP 1: Secrets Injection
# Load environment variables from the .env file
load_dotenv()
SERVER_SECRET = os.getenv("SECRET_API_KEY", "default_fallback_key")

app = Flask(__name__)

print("Loading DistilBERT model... This might take a moment on the first run.")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route('/predict', methods=['POST'])
def predict():
    # Example of using the injected secret for security
    provided_key = request.headers.get("x-api-key")
    if provided_key != SERVER_SECRET:
        return jsonify({"error": "Unauthorized. Invalid API Key."}), 401

    data = request.get_json(force=True)
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = classifier(text)[0]
    
    return jsonify({
        "input": text,
        "prediction": result['label'],
        "confidence": result['score']
    })

if __name__ == '__main__':
    print("Starting server on port 5000...")
    serve(app, host='0.0.0.0', port=5000)