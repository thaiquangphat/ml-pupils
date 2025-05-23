# HMM Training Script

import numpy as np
import pandas as pd
import joblib
from hmmlearn import hmm
from pathlib import Path
import time
from typing import Dict, Any, Tuple

from utils.logger import get_logger
from utils.utils import get_save_name
from models.hmm.build_features import process_images_to_features, COLS

logger = get_logger("hmm_train")

DEFAULT_HMM_ARGS = {
    "n_components": 4, # Default number of states, should match number of classes ideally
    "covariance_type": "diag",
    "n_iter": 100,
    "tol": 1e-3,
    "random_state": 42,
    "init_params": "stmc", # Initialize startprob, transmat, means, covars
    "params": "stmc"       # Estimate all parameters during training
}

DEFAULT_FEATURE_ARGS = {
    "output_dir": "features/output/hmm"
}

def train(
    dataset: Any, # Changed type hint from Tuple to Any, as it's an ImageDataset instance
    save_dir: Path,
    model_args: Dict[str, Any] = None,
    feature_args: Dict[str, Any] = None
) -> None:
    """Trains a separate Gaussian HMM for each class in the dataset.

    Args:
        dataset (ImageDataset): An instance of ImageDataset containing images and labels.
        save_dir (Path): Directory to save the trained HMM models.
        model_args (Dict[str, Any]): Arguments for the GaussianHMM model, loaded from config.
        feature_args (Dict[str, Any]): Arguments for feature extraction, loaded from config.
    """
    start_time = time.time()
    logger.info("Starting HMM training...")

    # Extract images and labels from the ImageDataset object
    images = dataset.images
    labels = dataset.labels
    
    # Update default args with provided ones
    hmm_params = DEFAULT_HMM_ARGS.copy()
    if model_args:
        hmm_params.update(model_args)
        
    feat_params = DEFAULT_FEATURE_ARGS.copy()
    if feature_args:
         feat_params.update(feature_args)

    # --- Feature Extraction --- 
    # Ensure feature output directory exists (optional, for saving intermediate features)
    # feature_output_dir = Path(feat_params.get("output_dir", "features/output/hmm"))
    # feature_output_dir.mkdir(parents=True, exist_ok=True)
    # feature_save_path = feature_output_dir / "train_features.csv"
    
    # Check if features already exist (optional optimization)
    # if feature_save_path.exists():
    #    logger.info(f"Loading pre-computed features from {feature_save_path}")
    #    df_features = pd.read_csv(feature_save_path)
    # else:
    df_features = process_images_to_features(images, labels, feat_params)
        # df_features.to_csv(feature_save_path, index=False)
        # logger.info(f"Saved computed features to {feature_save_path}")

    X_train = df_features[COLS].values
    y_train = df_features["label"].values
    
    # Check for NaN/inf values in features after extraction
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        logger.warning("NaN or Inf values found in features. Replacing with 0.")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # --- HMM Training per Class --- 
    unique_labels = np.unique(y_train)
    trained_models: Dict[Any, hmm.GaussianHMM] = {}
    logger.info(f"Training {len(unique_labels)} HMM models (one per class)...")

    for label_val in unique_labels:
        logger.info(f"Training HMM for class: {label_val}")
        # Select data for the current class
        X_class = X_train[y_train == label_val]
        
        if X_class.shape[0] == 0:
            logger.warning(f"No samples found for class {label_val}. Skipping HMM training for this class.")
            continue
        elif X_class.shape[0] < hmm_params["n_components"]:
             logger.warning(f"Class {label_val} has fewer samples ({X_class.shape[0]}) than n_components ({hmm_params['n_components']}). Reducing n_components for this class.")
             current_hmm_params = hmm_params.copy()
             current_hmm_params["n_components"] = max(1, X_class.shape[0]) # Ensure at least 1 component
        else:
             current_hmm_params = hmm_params

        # Initialize Gaussian HMM
        # Ensure n_components doesn't exceed number of samples for the class
        n_components_adjusted = min(current_hmm_params["n_components"], X_class.shape[0])
        if n_components_adjusted != current_hmm_params["n_components"]:
             logger.warning(f"Adjusted n_components to {n_components_adjusted} for class {label_val} due to limited samples.")
        
        model = hmm.GaussianHMM(
            n_components=n_components_adjusted, 
            covariance_type=current_hmm_params["covariance_type"],
            n_iter=current_hmm_params["n_iter"],
            tol=float(current_hmm_params["tol"]),
            random_state=current_hmm_params["random_state"],
            init_params=current_hmm_params["init_params"],
            params=current_hmm_params["params"]
        )
        
        try:
            # Fit the model to the class data
            # hmmlearn expects sequences, but for independent classification,
            # we can treat each sample as a sequence of length 1.
            # Providing X_class directly often works for GaussianHMM fit.
            model.fit(X_class)
            trained_models[label_val] = model
            logger.info(f"HMM training completed for class: {label_val}")
        except ValueError as e:
            logger.error(f"Error training HMM for class {label_val}: {e}")
            logger.error(f"Data shape: {X_class.shape}")
            # Skip this class if model fitting fails critically
            continue 
            
    # --- Save Models --- 
    if not trained_models:
        logger.error("No HMM models were trained successfully. Aborting save.")
        return
        
    save_name = get_save_name("hmm_models", "joblib")
    save_path = save_dir / save_name
    try:
        joblib.dump(trained_models, save_path)
        logger.info(f"HMM models saved successfully to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save HMM models to {save_path}: {e}")

    end_time = time.time()
    logger.info(f"HMM training finished in {end_time - start_time:.2f} seconds.")

    # --- Installation Note --- 
    try:
        import hmmlearn
    except ImportError:
        logger.warning("Package 'hmmlearn' not found. Please install it: pip install hmmlearn")
