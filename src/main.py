import argparse
from datetime import datetime
import os

from src.training.experiment import ClassificationExperiment
from src.models.classifiers import SimpleClassifier


def main():
    parser = argparse.ArgumentParser(description='Train a classification model on embeddings')
    parser.add_argument('--h5_path', type=str, required=True, help='Path to the HDF5 file')
    parser.add_argument('--medium', type=str, choices=['audio_embeddings', 'symbolic_embeddings'],
                      required=True, help='Type of embeddings to use')
    parser.add_argument('--label', type=str, required=True, help='Target label for classification')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, help='Directory to save model and results')
    parser.add_argument('--embedding_type', type=str, choices=['sequence', 'average'],
                      default='average', help='Type of embedding to use (sequence or average)')
    args = parser.parse_args()
    
    # Create save directory with timestamp if not provided
    if not args.save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"results/{args.medium}_{args.label}_{timestamp}"
    
    # Define model configuration based on embedding type
    if args.embedding_type == 'sequence':
        model_kwargs = {
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.5
        }
    else:
        model_kwargs = {
            'hidden_dims': [512, 256],
            'dropout': 0.5
        }
    
    # Create and run experiment
    experiment = ClassificationExperiment(
        h5_path=args.h5_path,
        medium=args.medium,
        label=args.label,
        model_fn=SimpleClassifier,  # This is not used anymore as model selection is handled in experiment
        model_kwargs=model_kwargs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        emb_type=args.embedding_type
    )
    
    # Train the model
    experiment.train(save_dir=args.save_dir)
    experiment.final_evaluation(save_dir=args.save_dir)


if __name__ == "__main__":
    main() 