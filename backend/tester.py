import tensorflow as tf
import numpy as np
from PIL import Image
import os

class CountryPredictor:
    def __init__(self, model_path='model-output/country_classifier_model.keras', class_mapping_path='model-output/country_classifier_model_classes.npy'):

        try:
            self.model = tf.keras.models.load_model(model_path)
        except:
            try:
                self.model = tf.keras.models.load_model(model_path, custom_objects={'TFSMLayer': tf.keras.layers.Layer})
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Please ensure the model file exists and is in the correct format (.keras or SavedModel)")
                raise
                
        try:
            # Load class mappings
            self.class_mapping = np.load(class_mapping_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading class mappings: {e}")
            print("Please ensure the class mapping file exists")
            raise
        
    def preprocess_image(self, image_path):
        try:
            # Load and preprocess the image
            img = Image.open(image_path).convert('RGB')  # Convert to RGB to ensure 3 channels
            img = img.resize((224, 224))  # Match the training size
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize
            img_array = tf.expand_dims(img_array, 0)
            return img_array
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def predict(self, image_path, top_k=3):
        # Preprocess the image
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return []
        
        try:
            # Make prediction
            predictions = self.model.predict(processed_image)
            
            # Handle different prediction formats
            if isinstance(predictions, dict):
                predictions = predictions[list(predictions.keys())[0]]
            
            # Ensure predictions is a numpy array
            predictions = np.array(predictions)
            if len(predictions.shape) == 2:
                predictions = predictions[0]
            
            # Get top k predictions, will be 3 
            top_indices = predictions.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                country = self.class_mapping[idx]
                confidence = predictions[idx]
                results.append({
                    'country': country,
                    'confidence': float(confidence)
                })
                
            return results
        except Exception as e:
            print(f"Error making prediction: {e}")
            return []

def main():
    try:
        # Initialize predictor
        print("Loading model...")
        predictor = CountryPredictor()
        
        # Example: predict a single image
        image_path = "testing_images/test3.png"
        if os.path.exists(image_path):
            results = predictor.predict(image_path)
            
            print(f"\nPredictions for {image_path}:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['country']}: {result['confidence']:.2%}")
        else:
            print(f"Image not found: {image_path}")
            
            # Example: predict multiple images in a directory
            test_dir = "testing_images/test3.png"
            if os.path.exists(test_dir):
                for image_file in os.listdir(test_dir):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(test_dir, image_file)
                        results = predictor.predict(image_path)
                        
                        print(f"\nPredictions for {image_file}:")
                        for i, result in enumerate(results, 1):
                            print(f"{i}. {result['country']}: {result['confidence']:.2%}")
            else:
                print(f"Test directory not found: {test_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()