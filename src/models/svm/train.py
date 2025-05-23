import joblib
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from typing import Dict, Any
import time

from utils.logger import get_logger
from utils.utils import get_save_name

logger = get_logger("svm_train")

DEFAULT_SVM_ARGS = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "probability": True,
    "random_state": 42
}

def train(dataset: Any, save_dir: Path, model_args: Dict[str, Any] = {}) -> None:
    """Trains an SVM model on the provided dataset object and saves it.

    Args:
        dataset (ImageDataset): An object containing the feature matrix (X)
                                  and the target vector (y) as attributes.
        save_dir (Path): The directory where the trained model should be saved.
        model_args (Dict[str, Any]): Dictionary of arguments for the SVM model.
                                     Supported keys: "C", "kernel", "gamma".
                                     Defaults are used if not provided.
    """
    start_time = time.time()
    logger.info("Starting SVM training...")

    # Extract data from the dataset object
    try:
        X = dataset.images # Access .images attribute
        y = dataset.labels # Access .labels attribute
    except AttributeError:
        logger.error("The provided dataset object does not have 'images' or 'labels' attributes.")
        raise TypeError("Dataset object must have 'images' and 'labels' attributes.")

    # Flatten the images if they are not already flat (e.g., (N, H, W, C) -> (N, H*W*C))
    # SVM expects 2D input: (n_samples, n_features)
    if X.ndim > 2:
        original_shape = X.shape
        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1)
        logger.info(f"Flattened image data from {original_shape} to {X.shape}")

    logger.info(f"Training dataset shape: X={X.shape}, y={y.shape}")

    # Ensure target vector is 1D
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
        logger.info("Flattened target vector to 1D")
    elif y.ndim > 1:
        logger.error(f"Target vector y has unexpected shape: {y.shape}")
        raise ValueError(f"Target vector y has unexpected shape: {y.shape}")

    # Update default args with provided ones
    svm_params = DEFAULT_SVM_ARGS.copy()
    if model_args:
        svm_params.update(model_args)
        logger.info(f"Updated model parameters: {svm_params}")

    logger.info(f"Initializing SVC model with parameters: {svm_params}")

    # Initialize and train the SVM model
    model = SVC(**svm_params)

    try:
        logger.info("Starting model training...")
        model.fit(X, y)
        logger.info("SVM model training completed successfully.")
        
        # Log model details
        n_support = model.n_support_
        logger.info(f"Number of support vectors per class: {n_support}")
        logger.info(f"Total number of support vectors: {sum(n_support)}")
        
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        raise

    # Save the model
    save_name = get_save_name("svm_model", "joblib")
    save_path = save_dir / save_name
    try:
        joblib.dump(model, save_path)
        logger.info(f"SVM model saved successfully to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save the model to {save_path}: {e}")
        raise

    end_time = time.time()
    logger.info(f"SVM training finished in {end_time - start_time:.2f} seconds.")

    # Installation Note
    try:
        import sklearn
        logger.info(f"Using scikit-learn version: {sklearn.__version__}")
    except ImportError:
        logger.warning("Package 'scikit-learn' not found. Please install it: pip install scikit-learn")
