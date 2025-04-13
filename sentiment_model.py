import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_sst2_data():
    """Load the SST-2 dataset"""
    # In a real project, you would download and process the SST-2 dataset
    # This is a simplified version for demonstration
    
    train_data = pd.read_csv('https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv', 
                            delimiter='\t', header=None)
    val_data = pd.read_csv('https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/dev.tsv', 
                          delimiter='\t', header=None)
    
    # Rename columns for clarity
    train_data.columns = ['sentence', 'label']
    val_data.columns = ['sentence', 'label']
    
    return train_data, val_data

def train_model(model, train_dataloader, val_dataloader, device):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Lists to store metrics
    train_losses = []
    val_accuracies = []
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Calculate average loss for this epoch
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f'Average training loss: {avg_train_loss}')
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_accuracy = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        val_accuracies.append(val_accuracy)
        
        print(f'Validation Accuracy: {val_accuracy}')
        print(f'Validation F1 Score: {val_f1}')
    
    return model, train_losses, val_accuracies

def visualize_results(train_losses, val_accuracies):
    """Create visualizations of training results"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', color='orange')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def main():
    # Load data
    train_data, val_data = load_sst2_data()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = SentimentDataset(
        texts=train_data['sentence'].values,
        labels=train_data['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    val_dataset = SentimentDataset(
        texts=val_data['sentence'].values,
        labels=val_data['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(device)
    
    # Train model
    model, train_losses, val_accuracies = train_model(
        model,
        train_dataloader,
        val_dataloader,
        device
    )
    
    # Save model
    model.save_pretrained('./sentiment_model')
    tokenizer.save_pretrained('./sentiment_model')
    
    # Visualize results
    visualize_results(train_losses, val_accuracies)
    
    print("Training complete! Model saved to ./sentiment_model")

if __name__ == "__main__":
    main()
