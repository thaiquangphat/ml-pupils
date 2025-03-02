import os
import cv2
import numpy as np
import torch
from skimage.filters import rank
from skimage.morphology import disk
from skimage.transform import rotate
from tqdm import tqdm

IMG_SIZE = (256, 256)

def cut_to_edge(img):
    """Crop the image to remove black edges (automatic bounding box)."""
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        img = img[y:y+h, x:x+w]
    
    return img

def apply_mask(img):
    """Apply a mask to extract the brain region (skull stripping)."""
    thresh = rank.otsu(img, disk(5))  # Otsu thresholding
    mask = thresh > 0
    img[~mask] = 0
    return img

def augment_image(img):
    """Apply random rotation and flipping (only for training)."""
    angle = np.random.uniform(-15, 15)  # Random rotation between -15 and 15 degrees
    img = rotate(img, angle, resize=False, mode="edge")
    if np.random.rand() > 0.5:
        img = np.fliplr(img)  # Random horizontal flip
    return img



def preprocess_mri(img):
    """Preprocess the image for Decision Tree or PyTorch"""
    # if (img.shape[-1] == 3):
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cut_to_edge(img)      
    img = apply_mask(img)       
    img = augment_image(img)
    img = cv2.resize(img, IMG_SIZE) 
    img = img / 255.0          
    return img
    
def preprocess_images(image_dir, img_size=(256, 256)):
    """
    Load all images in a directory and apply preprocessing.
    
    Parameters:
        image_dir: the directory include subdirectories corresponding to classes of images
        img_size: for resizing image. default (256,256)
    
    Return:
        tuple of (data, labels) in numpy.array format
    """
    data = []
    labels = []
    
    classes = [f for f in os.listdir(image_dir) if f != '.DS_Store']
    print(classes)
    
    for c, label in enumerate(classes):
        class_path = os.path.join(image_dir, label)
        
        for img_file in tqdm(os.listdir(class_path)):
            # Skip hidden files
            if img_file.startswith('.'):
                continue

            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if len(img.shape) == 3:  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            processed_img = preprocess_mri(img)
    
            data.append(processed_img)
            labels.append(c)  
    
    return np.array(data), np.array(labels)
    