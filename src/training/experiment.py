import json
import os
from typing import Dict, Any, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split

from src.data.embedding_dataset import EmbeddingDataset
from src.utils.metrics import evaluate_model


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
        train_val_split: float = 0.8,
        random_seed: int = 42
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
        self.train_val_split = train_val_split
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize dataset and model
        self._setup_dataset()
        self._setup_model()
        
    def _setup_dataset(self):
        """Initialize dataset and create train/val dataloaders."""
        # Create full dataset
        self.dataset = EmbeddingDataset(
            h5_path=self.h5_path,
            medium=self.medium,
            emb_type='average',  # Using average embeddings for classification
            label=self.label
        )
        
        # Split into train and validation sets
        train_size = int(self.train_val_split * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
    def _setup_model(self):
        """Initialize the model and training components."""
        # Get input dimension from first sample
        sample_emb, _ = self.dataset[0]
        input_dim = sample_emb.shape[0]
        num_classes = len(self.dataset.get_classes())
        
        # Initialize model
        self.model = self.model_fn(
            input_dim=input_dim,
            num_classes=num_classes,
            **self.model_kwargs
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
    
    def final_evaluation(self):
        """Perform final evaluation and print detailed metrics."""
        val_acc, val_preds, val_labels = evaluate_model(self.model, self.val_loader, self.device)
        print("\nFinal Validation Results:")
        print(f"Accuracy: {val_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            val_labels,
            val_preds,
            target_names=self.dataset.get_classes()
        )) 