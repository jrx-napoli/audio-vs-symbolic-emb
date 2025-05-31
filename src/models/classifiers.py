import torch.nn as nn


class SimpleClassifier(nn.Module):
    """A simple feed-forward neural network for classification."""
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float = 0.5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.classifier(x) 