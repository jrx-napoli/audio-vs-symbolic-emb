import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, loader: DataLoader, device: str) -> tuple:
    """
    Evaluate model and return accuracy and predictions.
    
    Parameters
    ----------
    model : nn.Module
        The model to evaluate
    loader : DataLoader
        DataLoader containing the evaluation data
    device : str
        Device to run evaluation on ('cuda' or 'cpu')
        
    Returns
    -------
    tuple
        (accuracy, predictions, true_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels 