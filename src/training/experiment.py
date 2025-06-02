import json
import os
from typing import Dict, Any, Optional, Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split, Dataset

from src.data.embedding_dataset import EmbeddingDataset
from src.models.classifiers import SimpleClassifier, LSTMClassifier
from src.utils.metrics import evaluate_model


def collate_sequences(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for sequence data that pads sequences to the same length.
    
    Args:
        batch: List of tuples (sequence, label)
        
    Returns:
        Tuple of (padded_sequences, labels)
    """
    # Separate sequences and labels
    sequences, labels = zip(*batch)
    
    # Pad sequences to the same length
    padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=True)
    
    # Convert labels to tensor
    labels = torch.tensor(labels)
    
    return padded_sequences, labels


class MappedDataset(Dataset):
    """Wrapper dataset that applies label mapping to another dataset."""
    def __init__(self, dataset: Dataset, label_mapping: Dict[int, int]):
        self.dataset = dataset
        self.label_mapping = label_mapping
        
    def __getitem__(self, idx):
        embedding, label = self.dataset[idx]
        # Convert tensor label to integer for mapping
        if torch.is_tensor(label):
            label = label.item()
        return embedding, self.label_mapping[label]
    
    def __len__(self):
        return len(self.dataset)
    
    def get_classes(self):
        return self.simplified_classes
    
    def set_simplified_classes(self, classes):
        self.simplified_classes = classes


class ClassificationExperiment:
    def __init__(
        self,
        h5_path: str,
        medium: str,
        label: str,
        model_fn: Callable[[int, int], nn.Module],
        model_kwargs: Dict[str, Any],
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        train_val_test_split: tuple = (0.6, 0.2, 0.2),  # (train, val, test) proportions
        random_seed: int = 42,
        emb_type: str = 'average'  # Add emb_type parameter with default
    ):
        self.h5_path = h5_path
        self.medium = medium
        self.label = label
        self.model_fn = model_fn
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        self.train_val_test_split = train_val_test_split
        self.random_seed = random_seed
        self.emb_type = emb_type
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize dataset and model
        self._setup_dataset()
        self._setup_model()
        
    def _setup_dataset(self):
        """Initialize dataset and create train/val/test dataloaders."""
        # Create full dataset
        original_dataset = EmbeddingDataset(
            h5_path=self.h5_path,
            medium=self.medium,
            emb_type=self.emb_type, 
            label=self.label
        )
        
        # Get original classes and create simplified class mapping
        original_classes = original_dataset.get_classes()
        simplified_classes = []
        class_mapping = {}  # Maps original class indices to new simplified indices
        
        # Create simplified class names and mapping
        for idx, class_name in enumerate(original_classes):
            simplified_name = class_name.split('/')[0]  # Take part before '/'
            if simplified_name not in simplified_classes:
                simplified_classes.append(simplified_name)
            class_mapping[idx] = simplified_classes.index(simplified_name)
        
        # Create wrapped dataset with mapping
        self.dataset = MappedDataset(original_dataset, class_mapping)
        self.dataset.set_simplified_classes(simplified_classes)
        
        # Print dataset information
        print("\nDataset Information:")
        print("-" * 50)
        print(f"Embedding type: {self.emb_type}")  
        print("Original classes mapped to simplified classes:")
        for idx, class_name in enumerate(original_classes):
            simplified_idx = class_mapping[idx]
            print(f"{class_name:<40} -> {simplified_classes[simplified_idx]}")
        print("\nSimplified Dataset Statistics:")
        print("-" * 50)
        print(f"Total number of samples: {len(self.dataset)}")
        print(f"Number of simplified classes: {len(simplified_classes)}")
        print("\nSimplified classes and their counts:")
        print("-" * 50)
        
        # Get labels for all samples using new mapping
        all_labels = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            all_labels.append(label)
            
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
            
        # Print class distribution
        for class_idx, class_name in enumerate(simplified_classes):
            count = label_counts.get(class_idx, 0)
            percentage = (count / len(self.dataset)) * 100
            print(f"{class_name:<30} {count:>5} samples ({percentage:>6.2f}%)")
        print("-" * 50)
        
        # Split into train, validation, and test sets
        total_size = len(self.dataset)
        train_size = int(self.train_val_test_split[0] * total_size)
        val_size = int(self.train_val_test_split[1] * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders with custom collate function
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_sequences if self.emb_type == 'sequence' else None  # Use collate_fn only for sequences
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_sequences if self.emb_type == 'sequence' else None
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_sequences if self.emb_type == 'sequence' else None
        )
        
    def _setup_model(self):
        """Initialize the model and training components."""
        # Get input dimension from first sample
        sample_emb, _ = self.dataset[0]
        input_dim = sample_emb.shape[-1]  # Last dimension is feature dimension
        num_classes = len(self.dataset.get_classes())
        
        # Initialize model based on embedding type
        if self.emb_type == 'sequence':
            # For sequence data, use LSTM classifier
            self.model = LSTMClassifier(
                input_dim=input_dim,
                hidden_dim=self.model_kwargs.get('hidden_dim', 256),
                num_classes=num_classes,
                num_layers=self.model_kwargs.get('num_layers', 2),
                dropout=self.model_kwargs.get('dropout', 0.5)
            ).to(self.device)
        else:
            # For averaged embeddings, use simple feed-forward classifier
            self.model = SimpleClassifier(
                input_dim=input_dim,
                hidden_dims=self.model_kwargs.get('hidden_dims', [512, 256]),
                num_classes=num_classes,
                dropout=self.model_kwargs.get('dropout', 0.5)
            ).to(self.device)
        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (embeddings, labels) in enumerate(self.train_loader):
            embeddings, labels = embeddings.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def train(self, save_dir: Optional[str] = None):
        """Train the model and optionally save results."""
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Evaluate
            val_acc, _, _ = evaluate_model(self.model, self.val_loader, self.device)
            val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")
            
            # Save best model
            if save_dir and val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                }, os.path.join(save_dir, 'best_model.pth'))
                
                # Save training history
                history = {
                    'train_losses': train_losses,
                    'val_accuracies': val_accuracies,
                    'best_val_accuracy': best_val_acc,
                    'num_epochs': self.num_epochs,
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'model_config': self.model_kwargs,
                    'classes': self.dataset.get_classes()
                }
                
                with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
                    json.dump(history, f, indent=4)
    
    def _plot_confusion_matrix(self, y_true, y_pred, classes, save_path=None):
        """Plot confusion matrix using seaborn."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def print_detailed_metrics(self, accuracies, preds, labels, dataset_name, classes):
        """Print detailed metrics for each class."""
        print(f"\nDetailed {dataset_name} Results:")
        print(f"Overall Accuracy: {accuracies:.4f}")
        
        # Calculate per-class metrics
        report = classification_report(
            labels,
            preds,
            target_names=classes,
            output_dict=True
        )
        
        print("\nPer-class Statistics:")
        print("-" * 80)
        print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 80)
        
        for class_name in classes:
            metrics = report[class_name]
            print(f"{class_name:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1-score']:>10.4f} {metrics['support']:>10d}")
        print("-" * 80)

    def final_evaluation(self, save_dir: Optional[str] = None):
        """Perform final evaluation on the test set and print detailed metrics."""
        classes = self.dataset.get_classes()
        
        # Evaluate on validation set
        val_acc, val_preds, val_labels = evaluate_model(self.model, self.val_loader, self.device)
        
        # Get unique classes present in validation data
        unique_val_classes = sorted(set(val_labels))
        val_class_names = [classes[i] for i in unique_val_classes]
        
        print("\nFinal Validation Results:")
        print(f"Accuracy: {val_acc:.4f}")
        print("\nValidation Classification Report:")
        print(classification_report(
            val_labels,
            val_preds,
            target_names=val_class_names,
            labels=unique_val_classes
        ))
        
        # Evaluate on test set
        test_acc, test_preds, test_labels = evaluate_model(self.model, self.test_loader, self.device)
        
        # Get unique classes present in test data
        unique_test_classes = sorted(set(test_labels))
        test_class_names = [classes[i] for i in unique_test_classes]
        
        print("\nFinal Test Results:")
        print(f"Accuracy: {test_acc:.4f}")
        print("\nTest Classification Report:")
        print(classification_report(
            test_labels,
            test_preds,
            target_names=test_class_names,
            labels=unique_test_classes
        ))
        
        if save_dir:
            # Save detailed metrics to JSON
            metrics = {
                'validation': {
                    'accuracy': float(val_acc),
                    'report': classification_report(
                        val_labels,
                        val_preds,
                        target_names=val_class_names,
                        labels=unique_val_classes,
                        output_dict=True
                    ),
                    'classes_present': val_class_names
                },
                'test': {
                    'accuracy': float(test_acc),
                    'report': classification_report(
                        test_labels,
                        test_preds,
                        target_names=test_class_names,
                        labels=unique_test_classes,
                        output_dict=True
                    ),
                    'classes_present': test_class_names
                },
                'all_possible_classes': classes
            }
            
            with open(os.path.join(save_dir, 'detailed_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print("\nNote: Some classes might not appear in validation/test sets due to random splitting.")
            print("All possible classes:", len(classes))
            print("Classes in validation set:", len(val_class_names))