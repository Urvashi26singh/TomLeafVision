import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 3  # Early Blight, Late Blight, Healthy
DATASET_PATH = 'dataset/plantvillage/tomato'  # Adjust based on your dataset structure
MODEL_SAVE_PATH = 'model/tomato_disease_model.h5'
CLASS_INDICES_PATH = 'model/class_indices.pkl'

# Create directories
os.makedirs('model', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('static', exist_ok=True)

def prepare_data():
    """
    Prepare data generators with augmentation for PlantVillage dataset
    """
    print("=" * 50)
    print("PREPARING PLANTVILLAGE DATASET")
    print("=" * 50)
    
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    print("\nLoading training data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        classes=['Early_blight', 'Late_blight', 'healthy']  # Adjust class names
    )
    
    # Load validation data
    print("\nLoading validation data...")
    validation_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        classes=['Early_blight', 'Late_blight', 'healthy']
    )
    
    # Save class indices
    class_indices = train_generator.class_indices
    with open(CLASS_INDICES_PATH, 'wb') as f:
        pickle.dump(class_indices, f)
    
    print(f"\nClasses: {class_indices}")
    print(f"Number of classes: {len(class_indices)}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    
    return train_generator, validation_generator, class_indices

def create_cnn_model(num_classes=3):
    """
    Create a CNN model for tomato disease detection
    """
    print("\n" + "=" * 50)
    print("CREATING CNN MODEL")
    print("=" * 50)
    
    # Option 1: Custom CNN (good for PlantVillage)
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_transfer_learning_model(num_classes=3):
    """
    Create a transfer learning model using pre-trained weights
    (Often better for PlantVillage dataset)
    """
    print("\n" + "=" * 50)
    print("CREATING TRANSFER LEARNING MODEL")
    print("=" * 50)
    
    # Load pre-trained VGG16 (you can also use ResNet50 or MobileNetV2)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def train_model():
    """
    Main training function
    """
    print("\n" + "=" * 50)
    print("PLANTVILLAGE TOMATO DISEASE DETECTION")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    # Prepare data
    train_generator, validation_generator, class_indices = prepare_data()
    
    # Choose model type (1 = Custom CNN, 2 = Transfer Learning)
    MODEL_TYPE = 2  # Transfer learning often works better
    
    if MODEL_TYPE == 1:
        model = create_cnn_model(len(class_indices))
    else:
        model, base_model = create_transfer_learning_model(len(class_indices))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Display model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=f'logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Train the model
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on validation data
    print("\n" + "=" * 50)
    print("EVALUATING MODEL")
    print("=" * 50)
    
    validation_generator.reset()
    evaluation = model.evaluate(validation_generator)
    metrics_names = ['Loss', 'Accuracy', 'Precision', 'Recall']
    
    for name, value in zip(metrics_names, evaluation):
        print(f"{name}: {value:.4f}")
    
    # Generate predictions for confusion matrix
    validation_generator.reset()
    predictions = model.predict(validation_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes
    class_labels = list(class_indices.keys())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))
    
    # Confusion matrix
    plot_confusion_matrix(true_classes, predicted_classes, class_labels)
    
    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    
    return model, history

def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('static/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(true_classes, predicted_classes, class_labels):
    """
    Plot confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_single_prediction(model, class_indices, image_path):
    """
    Test model on a single image
    """
    from PIL import Image
    
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get class name
    class_names = list(class_indices.keys())
    predicted_disease = class_names[predicted_class]
    
    print(f"\nPrediction: {predicted_disease}")
    print(f"Confidence: {confidence:.2%}")
    
    return predicted_disease, confidence

if __name__ == "__main__":
    # Train the model
    model, history = train_model()
    
    # Load class indices
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Model saved: {MODEL_SAVE_PATH}")
    print(f"Class indices: {class_indices}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # Optional: Test on a sample image
    # test_image = 'path/to/test/image.jpg'
    # if os.path.exists(test_image):
    #     test_single_prediction(model, class_indices, test_image)
    
    