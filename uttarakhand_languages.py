# uttarakhand_languages.py
# Application of sentiment analysis to indigenous languages of Uttarakhand

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load pre-trained model
MODEL_PATH = "./sentiment_model"  # Path to saved XLM-RoBERTa model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def create_sample_kumaoni_dataset():
    """
    Create a small sample dataset of Kumaoni sentences with sentiment labels.
    In a real project, you would collect or translate actual Kumaoni text.
    """
    kumaoni_samples = [
        # Positive examples
        {"text": "आज मेरो दिन बड़ो अच्छो रयो", "label": 1},  # Today was a very good day
        {"text": "ई किताब पढ़ने को बड़ो मजो आयो", "label": 1},  # Reading this book was very enjoyable
        {"text": "हमरा गाँव मा प्राकृतिक सौंदर्य बड़ो अच्छो छ", "label": 1},  # The natural beauty in our village is very nice
        {"text": "थ्वारी मदद के बड़ो धन्यवाद", "label": 1},  # Thank you very much for your help
        {"text": "हमरी संस्कृति बड़ी समृद्ध छ", "label": 1},  # Our culture is very rich
        
        # Negative examples
        {"text": "आजको मौसम बड़ो खराब छ", "label": 0},  # Today's weather is very bad
        {"text": "परीक्षा मा असफल होने से बड़ो दुख हुयो", "label": 0},  # I felt very sad after failing the exam
        {"text": "पहाड़न मा भूकंप से बड़ो नुकसान हुयो", "label": 0},  # The earthquake caused a lot of damage in the mountains
        {"text": "बाढ़ से फसलन को हानि हुयी", "label": 0},  # The crops were damaged by the flood
        {"text": "बिमारी के कारण काम ना कर सको", "label": 0},  # Could not work due to illness
    ]
    
    return pd.DataFrame(kumaoni_samples)

def create_sample_garhwali_dataset():
    """
    Create a small sample dataset of Garhwali sentences with sentiment labels.
    In a real project, you would collect or translate actual Garhwali text.
    """
    garhwali_samples = [
        # Positive examples
        {"text": "आज मेरू दिन बौत बधिया गॊ", "label": 1},  # Today my day was very good
        {"text": "द्यु री कृपा मेरा परिवार पर सदा छ", "label": 1},  # God's grace is always upon my family
        {"text": "हमारु गांव रु सौंदर्य बड़ु मनमोहक छ", "label": 1},  # The beauty of our village is very captivating
        {"text": "त्वाड़ी मदद बड़ी क्यूं लगी", "label": 1},  # Your help was very useful
        {"text": "हमारी भौं अर संस्कृति हमारौ गौरव छन", "label": 1},  # Our language and culture are our pride
        
        # Negative examples
        {"text": "पहाड़ों म भूकंप बड़ी तबाही मचाली", "label": 0},  # The earthquake caused great destruction in the mountains
        {"text": "बरखा न खेतौं कू बरबाद कर दिनू", "label": 0},  # The rain destroyed the fields
        {"text": "पलायन स गांव खाली होंदा जांदा", "label": 0},  # Villages are becoming empty due to migration
        {"text": "मेरी तबियत ठीक नी छ", "label": 0},  # I am not feeling well
        {"text": "मुश्किल स्थिति म फंस गेनू", "label": 0},  # Got stuck in a difficult situation
    ]
    
    return pd.DataFrame(garhwali_samples)

def predict_sentiment(text):
    """Predict sentiment for given text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=1)
        prediction = torch.argmax(scores, dim=1).item()
        confidence = scores[0][prediction].item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, prediction, confidence

def evaluate_on_dataset(dataset):
    """Evaluate model on a given dataset"""
    predictions = []
    confidences = []
    
    for _, row in dataset.iterrows():
        _, prediction, confidence = predict_sentiment(row['text'])
        predictions.append(prediction)
        confidences.append(confidence)
    
    accuracy = accuracy_score(dataset['label'], predictions)
    report = classification_report(dataset['label'], predictions)
    
    return accuracy, report, predictions, confidences

def visualize_results(kumaoni_results, garhwali_results):
    """Visualize performance on indigenous languages"""
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    languages = ['Kumaoni', 'Garhwali']
    accuracies = [kumaoni_results[0], garhwali_results[0]]
    plt.bar(languages, accuracies, color=['skyblue', 'lightgreen'])
    plt.title('Sentiment Analysis Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Confidence distribution for Kumaoni
    plt.subplot(2, 2, 3)
    sns.histplot(kumaoni_results[3], bins=10, kde=True, color='skyblue')
    plt.title('Confidence Distribution - Kumaoni')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    
    # Confidence distribution for Garhwali
    plt.subplot(2, 2, 4)
    sns.histplot(garhwali_results[3], bins=10, kde=True, color='lightgreen')
    plt.title('Confidence Distribution - Garhwali')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('uttarakhand_languages_results.png')
    plt.close()

def main():
    # Create sample datasets
    kumaoni_data = create_sample_kumaoni_dataset()
    garhwali_data = create_sample_garhwali_dataset()
    
    print("Evaluating on Kumaoni dataset...")
    kumaoni_results = evaluate_on_dataset(kumaoni_data)
    print(f"Kumaoni Accuracy: {kumaoni_results[0]:.4f}")
    print("Classification Report for Kumaoni:")
    print(kumaoni_results[1])
    
    print("\nEvaluating on Garhwali dataset...")
    garhwali_results = evaluate_on_dataset(garhwali_data)
    print(f"Garhwali Accuracy: {garhwali_results[0]:.4f}")
    print("Classification Report for Garhwali:")
    print(garhwali_results[1])
    
    # Visualize results
    visualize_results(kumaoni_results, garhwali_results)
    print("\nResults visualization saved to uttarakhand_languages_results.png")
    
    # Example of practical application
    print("\nDemonstrating practical application for indigenous language sentiment analysis...")
    
    kumaoni_example = "हमरो पहाड़ी क्षेत्र को विकास बड़ो जरूरी छ"  # Development of our hilly region is very important
    sentiment, _, confidence = predict_sentiment(kumaoni_example)
    print(f"Kumaoni text: '{kumaoni_example}'")
    print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")
    
    garhwali_example = "हमारी भाषा कू संरक्षण दिये जाण की जरूरत छ"  # Our language needs to be preserved
    sentiment, _, confidence = predict_sentiment(garhwali_example)
    print(f"Garhwali text: '{garhwali_example}'")
    print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()
