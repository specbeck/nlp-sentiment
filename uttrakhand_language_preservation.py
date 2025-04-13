# uttarakhand_language_preservation.py
# Application for indigenous language preservation in Uttarakhand

import torch
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import os

class IndigenousLanguagePreserver:
    def __init__(self, base_model="xlm-roberta-base"):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(base_model)
        self.model = XLMRobertaForMaskedLM.from_pretrained(base_model)
        
        # Dictionary mapping for special characters in Kumaoni/Garhwali not in standard Unicode
        self.special_char_mapping = {
            # Add special character mappings if needed
        }
    
    def preprocess_text(self, text):
        """Preprocess indigenous language text to handle special characters"""
        for original, replacement in self.special_char_mapping.items():
            text = text.replace(original, replacement)
        return text
    
    def collect_language_samples(self, filepath):
        """Collect language samples from a text file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.readlines()
    
    def create_parallel_corpus(self, indigenous_texts, hindi_translations):
        """Create a parallel corpus of indigenous language and Hindi translations"""
        return pd.DataFrame({
            'indigenous_text': indigenous_texts,
            'hindi_translation': hindi_translations
        })
    
    def fine_tune_for_language(self, corpus, output_dir):
        """Fine-tune model for indigenous language understanding"""
        # Code for fine-tuning would go here
        # This is a simplified placeholder
        print(f"Fine-tuning model on {len(corpus)} examples")
        print(f"Model would be saved to {output_dir}")
        
        # In a real implementation, we would:
        # 1. Create a dataset from the corpus
        # 2. Set up training arguments
        # 3. Fine-tune the model
        # 4. Save the model to output_dir
    
    def analyze_sentiment(self, text):
        """Analyze sentiment in indigenous language text"""
        # In a real implementation, this would use the fine-tuned model
        preprocessed = self.preprocess_text(text)
        # Placeholder sentiment analysis logic
        positive_words = ["अच्छो", "बधिया", "खुशी", "प्यारो", "सुंदर"]
        negative_words = ["बुरो", "दुख", "खराब", "मुश्किल", "हानि"]
        
        pos_count = sum(1 for word in positive_words if word in preprocessed)
        neg_count = sum(1 for word in negative_words if word in preprocessed)
        
        sentiment = "Positive" if pos_count > neg_count else "Negative" if neg_count > pos_count else "Neutral"
        confidence = abs(pos_count - neg_count) / (pos_count + neg_count + 1)
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_count': pos_count,
            'negative_count': neg_count
        }
    
    def document_dialect_variations(self, texts, region_labels):
        """Document variations in dialects across different regions of Uttarakhand"""
        dialect_df = pd.DataFrame({
            'text': texts,
            'region': region_labels
        })
        
        # Count unique words by region
        region_vocabularies = {}
        for region in set(region_labels):
            region_texts = dialect_df[dialect_df['region'] == region]['text'].tolist()
            combined_text = ' '.join(region_texts)
            words = combined_text.split()
            unique_words = set(words)
            region_vocabularies[region] = unique_words
        
        # Find shared and unique vocabulary
        all_regions = list(region_vocabularies.keys())
        shared_vocab = set.intersection(*[region_vocabularies[r] for r in all_regions])
        
        # Find words unique to each region
        unique_by_region = {}
        for region in all_regions:
            other_regions = [r for r in all_regions if r != region]
            other_vocab = set.union(*[region_vocabularies[r] for r in other_regions])
            unique_vocab = region_vocabularies[region] - other_vocab
            unique_by_region[region] = unique_vocab
        
        return {
            'region_vocabularies': region_vocabularies,
            'shared_vocabulary': shared_vocab,
            'unique_by_region': unique_by_region
        }
    
    def visualize_dialect_distribution(self, dialect_analysis):
        """Visualize dialect variations across regions"""
        plt.figure(figsize=(12, 8))
        
        # Vocabulary size by region
        regions = list(dialect_analysis['region_vocabularies'].keys())
        vocab_sizes = [len(dialect_analysis['region_vocabularies'][r]) for r in regions]
        
        plt.subplot(2, 1, 1)
        plt.bar(regions, vocab_sizes, color='skyblue')
        plt.title('Vocabulary Size by Region')
        plt.xlabel('Region')
        plt.ylabel('Number of Unique Words')
        
        # Unique words by region
        unique_counts = [len(dialect_analysis['unique_by_region'][r]) for r in regions]
        
        plt.subplot(2, 1, 2)
        plt.bar(regions, unique_counts, color='lightgreen')
        plt.title('Region-Specific Words')
        plt.xlabel('Region')
        plt.ylabel('Number of Unique Words')
        
        plt.tight_layout()
        plt.savefig('dialect_distribution.png')
        plt.close()
    
    def create_digital_dictionary(self, corpus, output_file):
        """Create a digital dictionary for the indigenous language"""
        all_words = set()
        for text in corpus['indigenous_text']:
            words = text.split()
            all_words.update(words)
        
        # Create word-definition pairs (in a real implementation, this would use translations)
        dictionary = {}
        for i, word in enumerate(all_words):
            if i < len(corpus):
                # Use Hindi translation as a placeholder definition
                sentence = corpus['indigenous_text'].iloc[i]
                translation = corpus['hindi_translation'].iloc[i]
                dictionary[word] = {
                    'definition': f"See translation: {translation}",
                    'example': sentence
                }
            else:
                dictionary[word] = {
                    'definition': "Definition not available",
                    'example': "Example not available"
                }
        
        # Save dictionary to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for word, info in sorted(dictionary.items()):
                f.write(f"{word}:\n")
                f.write(f"  Definition: {info['definition']}\n")
                f.write(f"  Example: {info['example']}\n\n")
        
        return dictionary
    
    def translate_text(self, text, source_lang, target_lang="hindi"):
        """Translate text between indigenous language and Hindi"""
        # In a real implementation, this would use a fine-tuned translation model
        # This is a placeholder implementation
        
        # Simple dictionary-based translation (very limited)
        kumaoni_to_hindi = {
            "अच्छो": "अच्छा",
            "बड़ो": "बहुत",
            "गाँव": "गांव",
            "पहाड़": "पहाड़",
            "दिन": "दिन"
        }
        
        garhwali_to_hindi = {
            "बौत": "बहुत",
            "गांव": "गांव",
            "बधिया": "अच्छा",
            "भौं": "भाषा",
            "द्यु": "ईश्वर"
        }
        
        hindi_to_kumaoni = {v: k for k, v in kumaoni_to_hindi.items()}
        hindi_to_garhwali = {v: k for k, v in garhwali_to_hindi.items()}
        
        translation_dict = None
        if source_lang == "kumaoni" and target_lang == "hindi":
            translation_dict = kumaoni_to_hindi
        elif source_lang == "garhwali" and target_lang == "hindi":
            translation_dict = garhwali_to_hindi
        elif source_lang == "hindi" and target_lang == "kumaoni":
            translation_dict = hindi_to_kumaoni
        elif source_lang == "hindi" and target_lang == "garhwali":
            translation_dict = hindi_to_garhwali
        
        if translation_dict:
            words = text.split()
            translated_words = [translation_dict.get(word, word) for word in words]
            return ' '.join(translated_words)
        else:
            return text  # Return original if translation not supported

def demo_uttarakhand_language_preservation():
    """Demonstrate the language preservation capabilities for Uttarakhand languages"""
    # Create language preserver
    preserver = IndigenousLanguagePreserver()
    
    # Sample Kumaoni texts with Hindi translations
    kumaoni_texts = [
        "हमरो पहाड़ी क्षेत्र को विकास बड़ो जरूरी छ",
        "हमरी भाषा हमरी पहचान छ",
        "आज मेरो दिन बड़ो अच्छो रयो",
        "कुमाऊंनी भाषा मा बड़ो ज्ञान छ",
        "पहाड़न मा लोक कलाओं को संरक्षण जरूरी छ"
    ]
    
    hindi_translations = [
        "हमारे पहाड़ी क्षेत्र का विकास बहुत जरूरी है",
        "हमारी भाषा हमारी पहचान है",
        "आज मेरा दिन बहुत अच्छा रहा",
        "कुमाऊंनी भाषा में बहुत ज्ञान है",
        "पहाड़ों में लोक कलाओं का संरक्षण जरूरी है"
    ]
    
    # Create parallel corpus
    corpus = preserver.create_parallel_corpus(kumaoni_texts, hindi_translations)
    print("Created parallel corpus with Kumaoni-Hindi texts")
    
    # Sample sentiment analysis
    print("\nSentiment Analysis Examples:")
    for text in kumaoni_texts[:2]:
        result = preserver.analyze_sentiment(text)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
    
    # Sample dialect variation analysis
    region_labels = ["Almora", "Nainital", "Pithoragarh", "Bageshwar", "Champawat"]
    # In a real implementation, we would have texts from each region
    sample_texts = kumaoni_texts * 5  # Duplicate for demonstration
    random_regions = [region_labels[i % len(region_labels)] for i in range(len(sample_texts))]
    
    dialect_analysis = preserver.document_dialect_variations(sample_texts, random_regions)
    print("\nDocumented dialect variations across regions")
    
    # Create digital dictionary
    dictionary = preserver.create_digital_dictionary(corpus, "kumaoni_dictionary.txt")
    print(f"\nCreated digital dictionary with {len(dictionary)} words")
    
    # Sample translation
    kumaoni_text = "हमरो गाँव बड़ो अच्छो छ"
    hindi_translation = preserver.translate_text(kumaoni_text, "kumaoni", "hindi")
    print(f"\nKumaoni: {kumaoni_text}")
    print(f"Hindi translation: {hindi_translation}")
    
    print("\nLanguage preservation demonstration complete")

if __name__ == "__main__":
    demo_uttarakhand_language_preservation()
