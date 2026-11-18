import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import random

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def build_autoencoder(input_shape=(224, 224, 3)):
    """Build convolutional autoencoder"""
    
    # Encoder
    input_img = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def load_batch(folder_path, batch_size=32, img_size=(224, 224)):
    """Load a random batch of images"""
    # Get all image files
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # Randomly select batch_size images
    selected_files = random.sample(all_files, min(batch_size, len(all_files)))
    
    batch_images = []
    for file in selected_files:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        
        if img is not None:
            img_resized = cv2.resize(img, img_size)
            img_normalized = img_resized / 255.0
            batch_images.append(img_normalized)
    
    return np.array(batch_images)

def data_generator(folder_path, batch_size=32):
    """Generator that yields (input, target) tuples for autoencoder"""
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    while True:
        # Randomly select batch
        selected = random.sample(all_files, min(batch_size, len(all_files)))
        
        batch = []
        for file in selected:
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            
            if img is not None:
                img_resized = cv2.resize(img, (224, 224))
                img_normalized = img_resized / 255.0
                batch.append(img_normalized)
        
        batch = np.array(batch)
        # For autoencoder, input and target are the same
        yield (batch, batch)  # This is the fix - return tuple

# Paths
train_folder = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\output\train_frames"

print("="*60)
print("TRAINING AUTOENCODER FOR INTRUSION DETECTION")
print("="*60)

# Count total frames
total_frames = len([f for f in os.listdir(train_folder) if f.endswith('.jpg')])
print(f"Total training frames available: {total_frames}")

# Build model
autoencoder = build_autoencoder()
autoencoder.summary()

# Create generator
batch_size = 32
train_gen = data_generator(train_folder, batch_size)

# Calculate steps per epoch
steps_per_epoch = min(50, total_frames // batch_size)
if steps_per_epoch == 0:
    steps_per_epoch = 10

print(f"\nTraining configuration:")
print(f"Batch size: {batch_size}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Number of epochs: 10")

# Train the model
print("\nStarting training...")
history = autoencoder.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    verbose=1
)

# Save the model
model_path = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\models"
os.makedirs(model_path, exist_ok=True)
autoencoder.save(os.path.join(model_path, 'autoencoder_model.h5'))
print(f"\n✅ Model saved to {model_path}")

# Save training history
import pickle
with open(os.path.join(model_path, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

print("\n✅ Training complete!")

# Plot training loss
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.savefig(os.path.join(model_path, 'training_loss.png'))
plt.show()