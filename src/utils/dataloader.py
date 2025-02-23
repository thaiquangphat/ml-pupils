import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import preprocess_images

IMG_SIZE = (256, 256)


def get_saved_numpy(save_path):
    """Load saved NumPy dataset if available."""
    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        return data['X'], data['y']
    return None, None

def save_numpy(save_path, X, y):
    """Save NumPy dataset."""
    np.savez_compressed(save_path, X=X, y=y)
    
class ImageDataset(Dataset):
    def __init__(self, image_dir, save_path, img_size=IMG_SIZE):
        """Initialize dataset: Load and preprocess images."""
        self.images, self.labels = preprocess_images(image_dir, img_size)
        save_numpy(save_path, self.images, self.labels)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_dataloader(image_dir, save_path=None, batch_size=64, img_size=IMG_SIZE, for_torch=False):
    """Returns DataLoader for ANN models or NumPy arrays for ML models.

    If save_path is provided and data exists, load it. Otherwise, create a new dataset and save it.
    """
    if save_path:
        X, y = get_saved_numpy(save_path)

        if X is not None and y is not None:
            print(f"Loaded saved dataset from {save_path}")
            if for_torch:
                dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X, dtype=torch.float32), 
                    torch.tensor(y, dtype=torch.long)
                )
                return DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return X, y

    # If no saved dataset found, preprocess images
    print("No saved dataset found. Creating new dataset...")
    dataset = ImageDataset(image_dir, save_path, img_size)
    print(f"Dataset saved to {save_path}")
    
    if for_torch:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return dataset.images, dataset.labels