import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15  # Fewer epochs for quick training
DATASET_PATH = 'dataset/train'

print("=" * 50)
print("QUICK TRAINING FOR LOCAL PLANTVILLAGE DATASET")
print("=" * 50)

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset path not found: {DATASET_PATH}")
    exit(1)

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True
)

# Load data
print("\nLoading dataset...")
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class indices
class_indices = train_generator.class_indices
with open('model/class_indices.pkl', 'wb') as f:
    pickle.dump(class_indices, f)

print(f"\nClasses: {class_indices}")

# Simple but effective model for PlantVillage
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_indices), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Train
print("\nStarting training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    verbose=1
)

# Save
model.save('model/tomato_disease_model.h5')
print(f"\n✅ Model saved to model/tomato_disease_model.h5")
print(f"✅ Class indices saved to model/class_indices.pkl")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2%}")