import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced constants
MODEL_NAME = "xlm-roberta-large"  # Upgraded from base to large
MAX_LEN = 160  # Increased from 128
BATCH_SIZE = 16
EPOCHS = 5  # Increased from 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01  # Added weight decay for regularization
WARMUP_RATIO = 0.1  # Added warmup steps
USE_FOCAL_LOSS = True  # Use focal loss for imbalanced data
FOCAL_GAMMA = 2.0  # Focal loss gamma parameter
SEED = 42  # For reproducibility
SAVE_INTERVAL = 0.5  # Save model every 0.5 epochs

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

class EnhancedSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        # Text augmentation for training set
        if self.augment and np.random.random() < 0.3:
            text = self._augment_text(text)
        
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
    
    def _augment_text(self, text):
        """Simple text augmentation techniques"""
        # Choose one augmentation technique randomly
        augmentation_type = np.random.choice(['swap', 'delete', 'synonym'])
        
        if augmentation_type == 'swap':
            # Randomly swap two words
            words = text.split()
            if len(words) > 1:
                idx1, idx2 = np.random.choice(range(len(words)), 2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
                return ' '.join(words)
        
        elif augmentation_type == 'delete':
            # Randomly delete a word
            words = text.split()
            if len(words) > 3:  # Ensure we don't delete too much
                idx = np.random.randint(0, len(words))
                words.pop(idx)
                return ' '.join(words)
        
        # Default: return original text if no augmentation applied
        return text

class FocalLoss(torch.nn.Module):
    """Focal Loss for dealing with class imbalance"""
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def load_and_preprocess_data():
    """Load and preprocess data with more options and better handling"""
    logger.info("Loading datasets...")
    
    # Load English dataset (SST-2)
    try:
        train_data = pd.read_csv('https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv', 
                                delimiter='\t', header=None)
        val_data = pd.read_csv('https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/dev.tsv', 
                              delimiter='\t', header=None)
        
        # Rename columns for clarity
        train_data.columns = ['sentence', 'label']
        val_data.columns = ['sentence', 'label']
        
        logger.info(f"Loaded {len(train_data)} training and {len(val_data)} validation samples")
        
        # Text preprocessing
        logger.info("Preprocessing text data...")
        
        # Remove special characters and extra spaces
        train_data['sentence'] = train_data['sentence'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        val_data['sentence'] = val_data['sentence'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        # Remove extra spaces
        train_data['sentence'] = train_data['sentence'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        val_data['sentence'] = val_data['sentence'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        
        # Check for class imbalance
        train_label_dist = Counter(train_data['label'])
        logger.info(f"Training label distribution: {train_label_dist}")
        
        return train_data, val_data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_data_loaders(train_data, val_data, tokenizer):
    """Create data loaders with stratified sampling and augmentation"""
    # Create datasets
    train_dataset = EnhancedSentimentDataset(
        texts=train_data['sentence'].values,
        labels=train_data['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        augment=True  # Enable augmentation for training
    )
    
    val_dataset = EnhancedSentimentDataset(
        texts=val_data['sentence'].values,
        labels=val_data['label'].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        augment=False  # No augmentation for validation
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
    
    return train_dataloader, val_dataloader

def train_enhanced_model(model, train_dataloader, val_dataloader, device):
    """Train the model with more advanced techniques and monitoring"""
    # Initialize optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    if USE_FOCAL_LOSS:
        loss_fn = FocalLoss(gamma=FOCAL_GAMMA)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    # Lists to store metrics
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    best_val_accuracy = 0
    
    logger.info(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        progress = 0
        
        for i, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Calculate loss
            if USE_FOCAL_LOSS:
                loss = loss_fn(outputs.logits, labels)
            else:
                loss = outputs.loss or loss_fn(outputs.logits, labels)
            
            epoch_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            # Progress tracking
            progress = (i + 1) / len(train_dataloader)
            if (i + 1) % (len(train_dataloader) // 5) == 0:
                logger.info(f"Training progress: {progress:.2%}, Loss: {loss.item():.4f}")
            
            # Save checkpoint every SAVE_INTERVAL epochs
            if progress >= SAVE_INTERVAL and progress - SAVE_INTERVAL < 1 / len(train_dataloader):
                checkpoint_path = f"./checkpoints/model_epoch{epoch+1}_step{i+1}.pt"
                os.makedirs("./checkpoints", exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Calculate average loss for this epoch
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_preds, val_true = [], []
        val_logits = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                val_logits.append(logits.cpu().numpy())
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_accuracy = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average='weighted')
        val_precision = precision_score(val_true, val_preds, average='weighted')
        val_recall = recall_score(val_true, val_preds, average='weighted')
        
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        
        logger.info(f"Validation Metrics:")
        logger.info(f"- Accuracy: {val_accuracy:.4f}")
        logger.info(f"- F1 Score: {val_f1:.4f}")
        logger.info(f"- Precision: {val_precision:.4f}")
        logger.info(f"- Recall: {val_recall:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), './best_model.pt')
            logger.info(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
    
    return model, {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'best_accuracy': best_val_accuracy
    }

def analyze_model_performance(model, val_dataloader, device, tokenizer):
    """Perform detailed analysis of model performance"""
    logger.info("Analyzing model performance...")
    
    model.eval()
    predictions = []
    true_labels = []
    texts = []
    all_logits = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            texts.extend(batch['text'])
            all_logits.extend(logits.cpu().numpy())
    
    # Convert logits to probabilities
    all_probs = torch.nn.functional.softmax(torch.tensor(all_logits), dim=1).numpy()
    
    # Analyze performance metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Analyze error cases
    error_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true != pred]
    error_analysis = [{
        'text': texts[i],
        'true_label': true_labels[i],
        'predicted': predictions[i],
        'confidence': all_probs[i][predictions[i]]
    } for i in error_indices]
    
    # Generate ROC curve data
    fpr, tpr, _ = roc_curve(true_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Generate Precision-Recall curve data
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, all_probs[:, 1])
    pr_auc = auc(recall_curve, precision_curve)
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix,
        'error_analysis': error_analysis,
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
        'pr_curve': {'precision': precision_curve, 'recall': recall_curve, 'auc': pr_auc}
    }
    
    # Log results
    logger.info(f"Final Model Performance:")
    logger.info(f"- Accuracy: {accuracy:.4f}")
    logger.info(f"- F1 Score: {f1:.4f}")
    logger.info(f"- Precision: {precision:.4f}")
    logger.info(f"- Recall: {recall:.4f}")
    logger.info(f"- ROC AUC: {roc_auc:.4f}")
    logger.info(f"- PR AUC: {pr_auc:.4f}")
    
    return results

def visualize_model_performance(training_history, analysis_results):
    """Create comprehensive visualizations of model performance"""
    logger.info("Creating performance visualizations...")
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create directory for visualizations
    os.makedirs('./visualizations', exist_ok=True)
    
    # 1. Training history plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(training_history['train_losses']) + 1), training_history['train_losses'], marker='o', linestyle='-', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(training_history['val_accuracies']) + 1), training_history['val_accuracies'], marker='o', linestyle='-', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(training_history['val_f1_scores']) + 1), training_history['val_f1_scores'], marker='o', linestyle='-', color='purple')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./visualizations/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = analysis_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
    plt.yticks([0.5, 1.5], ['Negative', 'Positive'])
    plt.tight_layout()
    plt.savefig('./visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(analysis_results['roc_curve']['fpr'], analysis_results['roc_curve']['tpr'], 
             color='darkorange', lw=2, label=f'ROC curve (area = {analysis_results["roc_curve"]["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./visualizations/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(analysis_results['pr_curve']['recall'], analysis_results['pr_curve']['precision'], 
             color='green', lw=2, label=f'PR curve (area = {analysis_results["pr_curve"]["auc"]:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./visualizations/pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Error analysis visualization
    if analysis_results['error_analysis']:
        error_df = pd.DataFrame(analysis_results['error_analysis'])
        error_df = error_df.sort_values('confidence', ascending=False).head(10)  # Top 10 high-confidence errors
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(error_df)), error_df['confidence'], color='salmon')
        plt.xlabel('Error Cases')
        plt.ylabel('Model Confidence')
        plt.title('Top 10 High-Confidence Errors')
        plt.xticks(range(len(error_df)), [f"{i+1}" for i in range(len(error_df))], rotation=45)
        plt.tight_layout()
        plt.savefig('./visualizations/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save error analysis details to CSV
        error_df.to_csv('./visualizations/error_analysis.csv', index=False)
    
    logger.info(f"Saved visualizations to ./visualizations/ directory")

def save_model_artifacts(model, tokenizer, config=None):
    """Save model, tokenizer, and configuration"""
    logger.info("Saving model artifacts...")
    
    # Create directories
    os.makedirs('./model_artifacts', exist_ok=True)
    
    # Save model
    model.save_pretrained('./model_artifacts/sentiment_model')
    
    # Save tokenizer
    tokenizer.save_pretrained('./model_artifacts/sentiment_model')
    
    # Save configuration information
    config_info = {
        'model_name': MODEL_NAME,
        'max_length': MAX_LEN,
        'training_epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'focal_loss_used': USE_FOCAL_LOSS,
        'focal_gamma': FOCAL_GAMMA if USE_FOCAL_LOSS else None,
    }
    
    import json
    with open('./model_artifacts/config.json', 'w') as f:
        json.dump(config_info, f, indent=4)
    
    logger.info(f"Model artifacts saved to ./model_artifacts/ directory")

def main():
    """Main function to train and evaluate the enhanced sentiment model"""
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    train_data, val_data = load_and_preprocess_data()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create data loaders
    train_dataloader, val_dataloader = create_data_loaders(train_data, val_data, tokenizer)
    
    # Initialize model with configuration
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(device)
    
    # Train model
    trained_model, training_history = train_enhanced_model(
        model,
        train_dataloader,
        val_dataloader,
        device
    )
    
    # Analyze model performance
    analysis_results = analyze_model_performance(trained_model, val_dataloader, device, tokenizer)
    
    # Visualize model performance
    visualize_model_performance(training_history, analysis_results)
    
    # Save model artifacts
    save_model_artifacts(trained_model, tokenizer, config)
    
    logger.info("Enhanced sentiment model training and evaluation complete!")

# Utils for cross-lingual transfer
def adapt_for_low_resource_language(base_model, tokenizer, target_lang_examples, device):
    """Adapt the model for a low-resource language using a small amount of data"""
    logger.info(f"Adapting model for low-resource language with {len(target_lang_examples)} examples")
    
    # Prepare data
    texts = [example['text'] for example in target_lang_examples]
    labels = [example['label'] for example in target_lang_examples]
    
    # Create dataset
    dataset = EnhancedSentimentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        augment=True  # Enable augmentation to create more variation from limited data
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # Smaller batch size for fine-tuning
        shuffle=True
    )
    
    # Fine-tuning settings
    optimizer = AdamW(base_model.parameters(), lr=1e-5)  # Lower learning rate for fine-tuning
    num_epochs = 10  # More epochs for fine-tuning on small data
    
    # Fine-tuning loop
    for epoch in range(num_epochs):
        base_model.train()
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            base_model.zero_grad()
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Adaptation epoch {epoch+1}/{num_epochs}: Loss {avg_loss:.4f}")
    
    # Save adapted model
    base_model.save_pretrained(f'./model_artifacts/adapted_model_{len(target_lang_examples)}_examples')
    logger.info(f"Adapted model saved")
    
    return base_model

if __name__ == "__main__":
    main()
