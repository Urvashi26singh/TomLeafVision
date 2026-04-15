import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import datetime

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30  # You can increase this for better accuracy
DATASET_PATH = 'dataset/train'  # Your local dataset path
MODEL_SAVE_PATH = 'model/tomato_disease_model.h5'
CLASS_INDICES_PATH = 'model/class_indices.pkl'

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)
os.makedirs('static', exist_ok=True)

print("=" * 60)
print("PLANTVILLAGE TOMATO DISEASE DETECTION TRAINING")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Dataset path: {os.path.abspath(DATASET_PATH)}")
print("=" * 60)

def check_dataset_structure():
    """Verify dataset structure"""
    required_classes = ['early_blight', 'late_blight', 'healthy']
    
    print("\nChecking dataset structure...")
    for class_name in required_classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        if os.path.exists(class_path):
            num_images = len([f for f in os.listdir(class_path) 
                            if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))])
            print(f"✓ {class_name}: {num_images} images")
        else:
            print(f"✗ {class_name} folder not found at: {class_path}")
            return False
    return True

def prepare_data():
    """
    Prepare data generators with augmentation for local PlantVillage dataset
    """
    print("\n" + "=" * 50)
    print("PREPARING DATASET")
    print("=" * 50)
    
    # Enhanced data augmentation for training
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
        validation_split=0.2  # 20% for validation
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    print("\nLoading training data from:", DATASET_PATH)
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = val_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Save class indices
    class_indices = train_generator.class_indices
    with open(CLASS_INDICES_PATH, 'wb') as f:
        pickle.dump(class_indices, f)
    
    print(f"\n✓ Classes found: {class_indices}")
    print(f"✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {validation_generator.samples}")
    
    return train_generator, validation_generator, class_indices

def create_model(num_classes=3):
    """
    Create a CNN model optimized for PlantVillage dataset
    Option 1: Custom CNN
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_transfer_model(num_classes=3):
    """
    Create transfer learning model (often better for PlantVillage)
    Option 2: MobileNetV2 Transfer Learning
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classifier
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    """
    Main training function
    """
    # Check dataset structure
    if not check_dataset_structure():
        print("\n❌ Dataset structure incorrect!")
        print("Please ensure your dataset is organized as:")
        print("  dataset/train/")
        print("    ├── early_blight/")
        print("    ├── late_blight/")
        print("    └── healthy/")
        return None, None
    
    # Prepare data
    train_generator, validation_generator, class_indices = prepare_data()
    
    # Choose model type (1 = Custom CNN, 2 = Transfer Learning)
    # Transfer learning (MobileNetV2) works better for PlantVillage
    USE_TRANSFER_LEARNING = True
    
    if USE_TRANSFER_LEARNING:
        print("\n" + "=" * 50)
        print("CREATING TRANSFER LEARNING MODEL (MobileNetV2)")
        print("=" * 50)
        model = create_transfer_model(len(class_indices))
    else:
        print("\n" + "=" * 50)
        print("CREATING CUSTOM CNN MODEL")
        print("=" * 50)
        model = create_model(len(class_indices))
    
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
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            'tomato_disease_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train the model
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)
    print(f"Training on {train_generator.samples} images")
    print(f"Validating on {validation_generator.samples} images")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print("=" * 50)
    
    history = model.fit(
        train_generator,
       
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator,
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
    evaluation = model.evaluate(validation_generator, verbose=1)
    metrics_names = ['Loss', 'Accuracy', 'Precision', 'Recall']
    
    print("\nFinal Validation Results:")
    for name, value in zip(metrics_names, evaluation):
        print(f"  {name}: {value:.4f}")
    
    # Save final model
    model.save('tomato_disease_model.h5')
    print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✓ Class indices saved to: {CLASS_INDICES_PATH}")
    
    return model, history

def plot_training_history(history):
    """
    Plot and save training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', color='blue')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', color='orange')
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision', color='blue')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', color='orange')
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall', color='blue')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', color='orange')
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = 'static/training_history.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to: {plot_path}")
    
    plt.show()

def test_prediction(model, class_indices, test_image_path=None):
    """
    Test the model on a sample image
    """
    from PIL import Image
    
    if test_image_path is None:
        # Find a sample image from validation set
        for class_name in class_indices.keys():
            class_path = os.path.join(DATASET_PATH, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    test_image_path = os.path.join(class_path, images[0])
                    true_class = class_name
                    break
    
    if test_image_path and os.path.exists(test_image_path):
        print("\n" + "=" * 50)
        print("TESTING SINGLE PREDICTION")
        print("=" * 50)
        
        # Load and preprocess image
        img = Image.open(test_image_path)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get class names
        class_names = list(class_indices.keys())
        predicted_class = class_names[predicted_class_idx]
        
        print(f"Test image: {os.path.basename(test_image_path)}")
        print(f"True class: {true_class if 'true_class' in locals() else 'Unknown'}")
        print(f"Predicted: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        
        # Show all class probabilities
        print("\nClass probabilities:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {predictions[0][i]:.2%}")
        
        return predicted_class, confidence

if __name__ == "__main__":
    try:
        # Train the model
        model, history = train_model()
        
        if model is not None:
            # Load class indices
            with open(CLASS_INDICES_PATH, 'rb') as f:
                class_indices = pickle.load(f)
            
            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Model: {MODEL_SAVE_PATH}")
            print(f"Classes: {class_indices}")
            
            if history:
                final_val_acc = history.history['val_accuracy'][-1]
                print(f"Final Validation Accuracy: {final_val_acc:.2%}")
            
            # Test on a sample image
            test_prediction(model, class_indices)
            
            print("\n" + "=" * 60)
            print("NEXT STEPS:")
            print("=" * 60)
            print("1. Run the Flask app: python app.py")
            print("2. Open browser: http://localhost:5000")
            print("3. Upload tomato leaf images for detection")
            
    except Exception as e:
        print("\n❌ Error during training:")
        print(f"   {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check if dataset folder exists: dataset/train/")
        print("2. Ensure subfolders exist: early_blight/, late_blight/, healthy/")
        print("3. Verify images are in JPG/PNG format")
        print("4. Check if you have at least 10 images per class")
