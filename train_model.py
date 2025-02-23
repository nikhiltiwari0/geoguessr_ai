import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
from pathlib import Path
import numpy as np

def create_model(num_classes, checkpoint_path=None):
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        return tf.keras.models.load_model(checkpoint_path)
    
    # Use ResNet50V2 as base model (you could also try others like EfficientNet)
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(image_dir, output_model_path):
    # Setup data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Load and prepare the training data
    train_generator = train_datagen.flow_from_directory(
        image_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    # Load and prepare the validation data
    validation_generator = train_datagen.flow_from_directory(
        image_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create or load model
    num_classes = len(train_generator.class_indices)
    checkpoint_path = output_model_path + '_checkpoint.keras'
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
        # Load training history if it exists
        history_path = f'{output_model_path}_history.npy'
        if os.path.exists(history_path):
            initial_history = np.load(history_path, allow_pickle=True).item()
            initial_epoch = len(initial_history['accuracy'])
        else:
            initial_epoch = 0
    else:
        print("Starting fresh training")
        model = create_model(num_classes)
        initial_epoch = 0
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_model_path + '_checkpoint.keras',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        # Optional: Add TensorBoard for visualization
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    # Train the model
    print("Training model...")
    history = model.fit(
        train_generator,
        epochs=20,
        initial_epoch=initial_epoch,  # Start from last epoch
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Save the model and class indices
    model.save(output_model_path + '.keras')
    
    # Save the class mapping
    class_indices = train_generator.class_indices
    class_mapping = {v: k for k, v in class_indices.items()}
    np.save(f'{output_model_path}_classes.npy', class_mapping)
    
    # Save training history to file
    history_dict = history.history
    np.save(f'{output_model_path}_history.npy', history_dict)
    
    return model, history, class_mapping

def main():
    # Get SageMaker environment variables
    training_dir = os.environ['SM_CHANNEL_TRAINING']
    model_dir = os.environ['SM_MODEL_DIR']
    
    print(f"Training directory: {training_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Contents of training directory: {os.listdir(training_dir)}")
    
    # Train the model
    model, history, class_mapping = train_model(training_dir, model_dir)
    
    # Print training results
    print("\nTraining completed!")
    print(f"Model saved to: {model_dir}")
    print(f"Number of countries classified: {len(class_mapping)}")
    
    # Print final metrics
    print("\nFinal training accuracy:", history.history['accuracy'][-1])
    print("Final validation accuracy:", history.history['val_accuracy'][-1])

if __name__ == "__main__":
    main() 