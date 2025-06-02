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


class LSTMClassifier(nn.Module):
    """LSTM-based classifier for sequence data."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # The output dimension will be 2*hidden_dim due to bidirectional LSTM
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use the last output from both directions
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden) 