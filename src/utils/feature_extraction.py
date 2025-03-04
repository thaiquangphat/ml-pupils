# src/utils/feature_extraction.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy

def segment_and_extract_features(img, is_tumor_image=True, min_area_threshold=10):
    features = {
        "tumor_present": 1 if is_tumor_image else 0,
        "area": 0.0, "perimeter": 0.0, "mean_intensity": 0.0,
        "eccentricity": 0.0, "solidity": 0.0, "contrast": 0.0, "homogeneity": 0.0,
        "segmentation_success": 0
    }

    if not is_tumor_image:
        return features

    img_uint8 = (img * 255).astype(np.uint8)
    main_region = None

    # Try multiple segmentation approaches
    # 1. Otsu Thresholding
    try:
        thresh = threshold_otsu(img_uint8)
        binary = (img_uint8 > thresh).astype(np.uint8)
        labeled = label(binary)
        regions = regionprops(labeled, intensity_image=img_uint8)
        main_region = max(regions, key=lambda r: r.area) if regions else None
    except:
        pass

    # 2. Adaptive Thresholding (fallback)
    if main_region is None or main_region.area < min_area_threshold:
        thresh = np.mean(img_uint8) * 0.8
        binary = (img_uint8 > thresh).astype(np.uint8)
        labeled = label(binary)
        regions = regionprops(labeled, intensity_image=img_uint8)
        main_region = max(regions, key=lambda r: r.area) if regions else None

    # If segmentation succeeds
    if main_region and main_region.area >= min_area_threshold:
        features["segmentation_success"] = 1
        features["area"] = float(main_region.area)
        features["perimeter"] = float(main_region.perimeter)
        features["mean_intensity"] = float(main_region.mean_intensity)
        features["eccentricity"] = float(main_region.eccentricity)
        features["solidity"] = float(main_region.solidity)

        roi = main_region.image * img_uint8[main_region.slice]
        glcm = graycomatrix(roi, [1], [0], levels=256, symmetric=True, normed=True)
        features["contrast"] = graycoprops(glcm, "contrast")[0, 0]
        features["homogeneity"] = graycoprops(glcm, "homogeneity")[0, 0]
    else:
        # Fallback: Extract global features
        features["mean_intensity"] = float(np.mean(img_uint8))
        features["contrast"] = float(np.std(img_uint8) / 128.0)
        hist, _ = np.histogram(img_uint8, bins=10, density=True)
        features["homogeneity"] = float(1.0 - entropy(hist + 1e-10) / np.log(10))
        features["area"] = float(img_uint8.size / 10)  # Approximation
        features["perimeter"] = float(4 * np.sqrt(img_uint8.size / 10))

    return features

def extract_features_from_npz(npz_file, output_path, split_name="train", chunk_size=1000):
    """Extract features from a processed NPZ file."""
    try:
        data = np.load(npz_file)
        X = data['X']
        y = data['y']
        class_names = ["notumor", "glioma", "meningioma", "pituitary"]
        
        print(f"Processing {len(X)} images from {npz_file}")
        features_list = []
        chunk_counter = 0
        
        for idx in tqdm(range(len(X)), desc="Extracting features", unit="img"):
            # Reshape if flattened
            img = X[idx]
            if len(img.shape) == 1:
                size = int(np.sqrt(img.shape[0]))  # Assuming square images
                img = img.reshape(size, size)
            
            # Get class name
            label_idx = y[idx]
            label = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
            
            # Extract features
            is_tumor = (label != "notumor")
            feats = segment_and_extract_features(img, is_tumor)
            feats["label"] = label
            feats["split"] = split_name
            feats["file_name"] = f"image_{idx}.npy"  # Synthetic filename
            features_list.append(feats)
            
            # Save in chunks to manage memory
            if len(features_list) >= chunk_size or idx == len(X) - 1:
                df_chunk = pd.DataFrame(features_list)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                mode = 'w' if chunk_counter == 0 else 'a'
                header = True if chunk_counter == 0 else False
                df_chunk.to_csv(output_path, mode=mode, header=header, index=False)

                features_list = []
                chunk_counter += 1
        
        return True
    
    except KeyError as e:
        print(f"Error loading {npz_file}: {str(e)}")
        # Create empty output file to avoid further errors
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame().to_csv(output_path, index=False)
        return False
    
def process_dataset(dataset="dataset/processed", output="feature_output", chunk_size=1000):
    """Process both train and test .npz files."""
    
    if os.path.exists(dataset):
        extract_features_from_npz(dataset, output, "train", chunk_size)
    else:
        print(f"Training file {dataset} not found.")

if __name__ == "__main__":
    process_dataset(dataset="../dataset/processed", output_dir="../output", chunk_size=1000)

