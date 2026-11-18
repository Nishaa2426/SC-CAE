import cv2
import os
import numpy as np
from tqdm import tqdm

def load_preprocessed_data(data_folder, img_size=(224, 224)):
    """Load and preprocess images on the fly"""
    images = []
    filenames = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
    
    print(f"Loading {len(filenames)} images from {os.path.basename(data_folder)}")
    
    for filename in tqdm(filenames[:1000]):  # Start with 1000 samples for testing
        img_path = os.path.join(data_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, img_size)
            img_normalized = img_resized / 255.0
            images.append(img_normalized)
    
    return np.array(images)

# Load training and testing data
train_folder = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\output\train_frames"
test_folder = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\output\test_frames"

print("Loading training data...")
X_train = load_preprocessed_data(train_folder)

print("\nLoading testing data...")
X_test = load_preprocessed_data(test_folder)

print(f"\n✅ Data loaded successfully!")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Save as numpy array (optional, if you have space)
output_file = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\output\dataset.npz"
np.savez(output_file, X_train=X_train, X_test=X_test)
print(f"\nDataset saved to: {output_file}")