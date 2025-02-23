import tensorflow as tf
from pathlib import Path
import os

# Import the training function from your existing code
from train_model import train_model

def main():
    # Set paths
    training_dir = Path('training-data')  # Your local training data directory
    model_dir = Path('model-output')      # Where to save the model
    
    # Create output directory if it doesn't exist
    model_dir.mkdir(exist_ok=True)
    
    print(f"Training directory: {training_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Contents of training directory: {os.listdir(training_dir)}")
    
    # Train the model
    model, history, class_mapping = train_model(training_dir, str(model_dir / 'country_classifier_model'))
    
    # Print final metrics
    print("\nTraining completed!")
    print(f"Model saved to: {model_dir}")
    print(f"Number of countries classified: {len(class_mapping)}")
    print("\nFinal training accuracy:", history.history['accuracy'][-1])
    print("Final validation accuracy:", history.history['val_accuracy'][-1])

if __name__ == "__main__":
    main()