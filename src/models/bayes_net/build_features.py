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

# In this function, we extract features from the input image. The method is GLCM-based, which is a common approach for texture analysis.
# For more information on GLCM, see: https://en.wikipedia.org/wiki/Gray-level_co-occurrence_matrix

def segment_and_extract_features(img, is_tumor_image=True, min_area_threshold=10):
    features = {
        "area": 0.0, "perimeter": 0.0,
        "eccentricity": 0.0, "solidity": 0.0, "contrast": 0.0, "homogeneity": 0.0,
        "energy": 0.0, "correlation": 0.0
    }

    if not is_tumor_image:
        return features

    img_float = img.astype(float)
    
    # Apply histogram equalization for better contrast
    img_eq = equalize_hist(img_float)
    
    # Convert to uint8 for processing
    img_uint8 = (img_eq * 255).astype(np.uint8)
    
    # Apply Gaussian smoothing to reduce noise
    img_smooth = gaussian(img_uint8, sigma=0.8)
    img_smooth_uint8 = (img_smooth * 255).astype(np.uint8)
    
    main_region = None
    
    # Try multiple improved segmentation approaches
    segmentation_methods = [
        # Method 1: Enhanced Otsu with morphological operations
        lambda img: binary_closing(binary_opening(
            img > threshold_otsu(img), disk(2)), disk(2)),
            
        # Method 2: Adaptive thresholding with local context
        lambda img: img > (np.mean(img) * 0.75),
        
        # Method 3: Watershed segmentation
        lambda img: watershed(ndimage.distance_transform_edt(img > np.mean(img)),
                              mask=img > np.mean(img) * 0.5)
    ]
    
    for segment_method in segmentation_methods:
        binary = segment_method(img_smooth_uint8)
        binary = clear_border(binary)  # Remove border artifacts
        labeled = label(binary)
        regions = regionprops(labeled, intensity_image=img_uint8)
        
        if regions:
            # Get the largest region as main tumor candidate
            main_region = max(regions, key=lambda r: r.area)
            if main_region.area >= min_area_threshold:
                break

    
    # If segmentation succeeds (main region found)
    if main_region and main_region.area >= min_area_threshold:
        
        # Basic shape features
        features["area"] = float(main_region.area)
        features["perimeter"] = float(main_region.perimeter)
        features["eccentricity"] = float(main_region.eccentricity)
        features["solidity"] = float(main_region.solidity)
        
        # Extract ROI
        roi = main_region.image * img_uint8[main_region.slice]
        
        if roi.size > 0:
            distances = [1, 3]  # Consider pixels at distances 1 and 3
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 degrees

            roi_valid = roi.copy()
            if np.sum(roi_valid) == 0:
                roi_valid[0, 0] = 1  # Ensure at least one non-zero value
            
            glcm = graycomatrix(roi_valid, distances, angles, levels=256, 
                                symmetric=True, normed=True)
            
            # Extract multiple GLCM properties
            features["contrast"] = float(np.mean(graycoprops(glcm, "contrast")))
            features["homogeneity"] = float(np.mean(graycoprops(glcm, "homogeneity")))
            features["energy"] = float(np.mean(graycoprops(glcm, "energy")))
            features["correlation"] = float(np.mean(graycoprops(glcm, "correlation")))
    else:
        # Fallback: Extract global features
        features["contrast"] = float(np.std(img_uint8) / 128.0)
        hist, _ = np.histogram(img_uint8, bins=10, density=True)
        features["homogeneity"] = float(1.0 - entropy(hist + 1e-10) / np.log(10))
        features["area"] = float(img_uint8.size / 10)  # Approximation
        features["perimeter"] = float(4 * np.sqrt(img_uint8.size / 10))
        
        # Additional global statistical features
        features["energy"] = float(np.sum(img_uint8**2) / (img_uint8.size * 255**2))
        features["correlation"] = 0.0  # Placeholder for global correlation

    return features