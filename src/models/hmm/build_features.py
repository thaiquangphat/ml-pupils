import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from scipy.stats import entropy
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import disk, binary_closing, binary_opening
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import watershed, clear_border
from skimage.exposure import equalize_hist
from typing import Dict, Any

# List of features extracted
COLS = ['area', 'perimeter', 'eccentricity', 'solidity', 'contrast', 'homogeneity', 'energy', 'correlation']


def segment_and_extract_features(
    img: np.ndarray, 
    is_tumor_image: bool = True, 
    min_area_threshold: int = 10
) -> Dict[str, float]:
    """Segments the image and extracts shape and texture features.

    This function attempts to segment the main region (presumably a tumor if is_tumor_image is True)
    using various methods and then calculates geometric and GLCM (Gray-Level Co-occurrence Matrix) 
    texture features. If segmentation fails or it's not a tumor image, default or global features 
    are returned.

    Args:
        img (np.ndarray): The input grayscale image (typically 2D).
        is_tumor_image (bool): Flag indicating if the image is expected to contain a tumor.
                                Affects whether segmentation is attempted.
        min_area_threshold (int): The minimum area for a segmented region to be considered valid.

    Returns:
        Dict[str, float]: A dictionary containing the extracted feature names and their values.
    """
    features: Dict[str, float] = {
        "area": 0.0, "perimeter": 0.0,
        "eccentricity": 0.0, "solidity": 0.0, "contrast": 0.0, "homogeneity": 0.0,
        "energy": 0.0, "correlation": 0.0
    }

    # If it's known not to be a tumor image, return default zero features
    if not is_tumor_image:
        return features

    # Ensure image is float for processing, handle potential type issues
    if img.dtype != np.uint8:
        img = img.astype(np.float32) # Convert to float for calculations
        # Normalize if not already in [0, 1] range approximately
        if img.max() > 1.5: 
             img = img / 255.0
    else:
        img = img.astype(np.float32) / 255.0

    # Apply histogram equalization for better contrast
    img_eq = equalize_hist(img)
    
    # Convert to uint8 for libraries expecting that range
    img_uint8 = (img_eq * 255).astype(np.uint8)
    
    # Apply Gaussian smoothing to reduce noise
    # Sigma can be adjusted; smaller values preserve more detail
    img_smooth = gaussian(img_uint8, sigma=0.8)
    img_smooth_uint8 = (img_smooth * 255).astype(np.uint8)
    
    main_region: Any = None # Using Any because regionprops return type is complex
    
    # Try multiple segmentation approaches to increase robustness
    segmentation_methods = [
        # Method 1: Enhanced Otsu with morphological operations
        # Good for images with bimodal histograms
        lambda im: binary_closing(binary_opening(
            im > threshold_otsu(im), disk(2)), disk(2)),
            
        # Method 2: Simple adaptive threshold based on mean intensity
        # Works if tumor is significantly brighter/darker than background mean
        lambda im: im > (np.mean(im) * 0.75),
        
        # Method 3: Watershed segmentation based on distance transform
        # Can separate touching objects but sensitive to noise/oversegmentation
        # lambda im: watershed(ndimage.distance_transform_edt(im > np.mean(im)),
        #                       mask=im > np.mean(im) * 0.5)
    ]
    
    for segment_method in segmentation_methods:
        try:
            binary = segment_method(img_smooth_uint8)
            binary = clear_border(binary)  # Remove regions touching the border
            labeled = label(binary)
            regions = regionprops(labeled, intensity_image=img_uint8)
            
            if regions:
                # Select the largest region as the primary candidate
                regions_filtered = [r for r in regions if r.area >= min_area_threshold]
                if regions_filtered:
                     main_region = max(regions_filtered, key=lambda r: r.area)
                     break # Found a suitable region, stop trying methods
        except Exception as e:
            # Log error if needed: print(f"Segmentation method failed: {e}")
            continue # Try next method

    # If segmentation succeeded and found a region above the threshold
    if main_region: 
        
        # --- Geometric Features --- 
        features["area"] = float(main_region.area)
        features["perimeter"] = float(main_region.perimeter)
        # Handle potential division by zero or invalid values if region is small/thin
        features["eccentricity"] = float(main_region.eccentricity) if main_region.major_axis_length > 0 else 0.0
        features["solidity"] = float(main_region.solidity) if main_region.convex_area > 0 else 0.0
        
        # --- Texture Features (GLCM) --- 
        # Extract the Region of Interest (ROI) defined by the segmentation
        minr, minc, maxr, maxc = main_region.bbox
        roi = img_uint8[minr:maxr, minc:maxc] * main_region.image # Apply mask

        # Ensure ROI is valid for GLCM (needs variance)
        if roi.size > 0 and np.ptp(roi) > 0: # ptp checks if range > 0
            distances = [1, 3]  # Pixel neighbor distances
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Directions
            
            # Calculate GLCM
            glcm = graycomatrix(roi, distances, angles, levels=256, 
                                symmetric=True, normed=True)
            
            # Aggregate GLCM properties (mean over distances/angles)
            features["contrast"] = float(np.mean(graycoprops(glcm, "contrast")))
            features["homogeneity"] = float(np.mean(graycoprops(glcm, "homogeneity")))
            features["energy"] = float(np.mean(graycoprops(glcm, "ASM"))) # ASM is equivalent to energy in some contexts
            features["correlation"] = float(np.mean(graycoprops(glcm, "correlation")))
        else:
            # If ROI is uniform or empty, texture features might be zero or undefined
            # Keep defaults (0.0) or assign specific fallback values if needed
            pass 
            
    # --- Fallback (if no valid region found) --- 
    # Optionally calculate global features if segmentation fails
    # else:
        # features["contrast"] = float(np.std(img_uint8))
        # ... (calculate other global features if desired) 

    # --- Final Check for NaN/inf --- 
    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            features[key] = 0.0 # Replace invalid numbers with 0.0
            
    return features

def process_images_to_features(
    images: np.ndarray, 
    labels: np.ndarray,
    feature_args: Dict[str, Any] # Placeholder for future feature extraction args
) -> pd.DataFrame:
    """Processes a batch of images to extract features for each.
    
    Args:
        images (np.ndarray): Array of images (n_samples, height, width).
        labels (np.ndarray): Array of labels (n_samples,).
        feature_args (Dict[str, Any]): Dictionary of arguments for feature extraction.

    Returns:
        pd.DataFrame: DataFrame where rows are samples and columns are features + label.
    """
    all_features = []
    print(f"Extracting features from {len(images)} images...")
    for i in tqdm(range(len(images)), desc="Feature Extraction"):
        img = images[i]
        label_val = labels[i]
        # Determine if it's a tumor image based on label (assuming 0 is non-tumor)
        is_tumor = label_val != 0 
        # Extract features for the current image
        img_features = segment_and_extract_features(img, is_tumor_image=is_tumor)
        img_features["label"] = label_val 
        all_features.append(img_features)
        
    # Convert list of dictionaries to DataFrame
    df_features = pd.DataFrame(all_features)
    
    # Reorder columns to have label last (optional, for consistency)
    cols_order = COLS + ["label"]
    df_features = df_features[cols_order]
    
    return df_features 