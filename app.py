# app.py - Backend API for sentiment analysis
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load pre-trained model
MODEL_PATH = "./sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Get text from request
    data = request.json
    text = data.get('text', '')
    
    # Tokenize text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=1)
        prediction = torch.argmax(scores, dim=1).item()
        confidence = scores[0][prediction].item()
    
    # Map prediction to sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'raw_scores': scores[0].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
