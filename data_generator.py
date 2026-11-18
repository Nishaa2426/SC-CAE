import cv2
import numpy as np
import os
from tensorflow.keras.utils import Sequence

class FrameDataGenerator(Sequence):
    """Generates batches of frames for training"""
    
    def __init__(self, folder_path, batch_size=32, img_size=(224, 224), shuffle=True):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        
        # Get all frame filenames
        self.filenames = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        self.indexes = np.arange(len(self.filenames))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.filenames) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.filenames))
        batch_indices = self.indexes[start_idx:end_idx]
        
        # Load batch images
        batch_images = []
        for idx in batch_indices:
            img_path = os.path.join(self.folder_path, self.filenames[idx])
            img = cv2.imread(img_path)
            
            if img is not None:
                # Resize and normalize
                img_resized = cv2.resize(img, self.img_size)
                img_normalized = img_resized / 255.0
                batch_images.append(img_normalized)
            else:
                # If image loading fails, add black image
                batch_images.append(np.zeros((*self.img_size, 3)))
        
        return np.array(batch_images)
    
    def on_epoch_end(self):
        """Shuffle after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Test the generator
if __name__ == "__main__":
    train_folder = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\output\train_frames"
    test_folder = r"C:\Users\snish\OneDrive\Desktop\SCNA Tag\intrusion_surveillance\output\test_frames"
    
    # Create generators
    train_generator = FrameDataGenerator(train_folder, batch_size=32)
    test_generator = FrameDataGenerator(test_folder, batch_size=32)
    
    print(f"✅ Data generators created successfully!")
    print(f"Training batches: {len(train_generator)}")
    print(f"Testing batches: {len(test_generator)}")
    
    # Test a batch
    sample_batch = train_generator[0]
    print(f"Sample batch shape: {sample_batch.shape}")
    print(f"Memory per batch: {sample_batch.nbytes / 1024 / 1024:.2f} MB")