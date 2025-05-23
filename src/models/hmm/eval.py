# HMM Evaluation Script 

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import time
from typing import Dict, Any, Tuple, Optional, List

from utils.logger import get_logger
from models.hmm.build_features import process_images_to_features, COLS
from hmmlearn import hmm # Import base HMM class for type hinting
from utils.testutils import metric_results # Added import

logger = get_logger("hmm_eval")

# Use the same default feature args as training
DEFAULT_FEATURE_ARGS = {
    "output_dir": "features/output/hmm"
}

def evaluate(
    dataset: Any, # Changed type hint from Tuple to Any
    saved_path: Path,
    model_args: Dict[str, Any] = None, # Keep for compatibility, maybe used later
    feature_args: Dict[str, Any] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Evaluates the trained HMM models on the test dataset.

    Loads a dictionary of HMM models (one per class) and predicts the class 
    for each test sample based on which model assigns the highest log-likelihood.

    Args:
        dataset (ImageDataset): Tuple containing test images (X) and labels (y).
        saved_path (Path): Path to the saved joblib file containing the dictionary of trained HMMs.
        model_args (Dict[str, Any]): Model arguments (currently unused in eval but kept for consistency).
        feature_args (Dict[str, Any]): Arguments for feature extraction.

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]: 
            A tuple containing: 
            - y_true: The true labels.
            - y_pred: The predicted labels.
            - y_scores: None, as HMM likelihoods are not directly probabilities for AUC.
    """
    start_time = time.time()
    logger.info(f"Starting HMM evaluation using model from: {saved_path}")

    if not saved_path.is_file():
        logger.error(f"Saved model file not found at {saved_path}")
        raise FileNotFoundError(f"No model file found at {saved_path}")

    # Extract images and true labels from the ImageDataset object
    images = dataset.images
    y_true = dataset.labels
    
    # Load the dictionary of trained HMM models
    try:
        trained_models: Dict[Any, hmm.GaussianHMM] = joblib.load(saved_path)
        logger.info(f"Successfully loaded {len(trained_models)} HMM models.")
    except Exception as e:
        logger.error(f"Failed to load models from {saved_path}: {e}")
        raise
        
    model_classes = list(trained_models.keys())
    if not model_classes:
         logger.error("Loaded model dictionary is empty. Cannot evaluate.")
         # Return empty arrays or raise error? Returning empty for now.
         return y_true, np.array([]), None

    # Update default feature args with provided ones
    feat_params = DEFAULT_FEATURE_ARGS.copy()
    if feature_args:
         feat_params.update(feature_args)
         
    # --- Feature Extraction --- 
    # feature_output_dir = Path(feat_params.get("output_dir", "features/output/hmm"))
    # feature_output_dir.mkdir(parents=True, exist_ok=True)
    # feature_save_path = feature_output_dir / "test_features.csv"

    # if feature_save_path.exists():
    #     logger.info(f"Loading pre-computed test features from {feature_save_path}")
    #     df_features = pd.read_csv(feature_save_path)
    # else:
    df_features = process_images_to_features(images, y_true, feat_params)
        # df_features.to_csv(feature_save_path, index=False)
        # logger.info(f"Saved computed test features to {feature_save_path}")

    X_test = df_features[COLS].values
    
    # Check for NaN/inf values
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        logger.warning("NaN or Inf values found in test features. Replacing with 0.")
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Prediction --- 
    y_pred = []
    logger.info(f"Predicting classes for {X_test.shape[0]} test samples...")

    for i in range(X_test.shape[0]):
        sample = X_test[i].reshape(1, -1) # Reshape for single sample prediction
        log_likelihoods: List[float] = []
        
        # Calculate log-likelihood for the sample under each class model
        for cls_label in model_classes:
            try:
                score = trained_models[cls_label].score(sample)
                log_likelihoods.append(score)
            except Exception as e:
                # Handle cases where scoring might fail (e.g., model poorly trained)
                logger.warning(f"Could not score sample {i} with model for class {cls_label}: {e}. Assigning -inf likelihood.")
                log_likelihoods.append(-np.inf) # Assign very low likelihood

        # Predict the class with the maximum log-likelihood
        if not log_likelihoods or all(ll == -np.inf for ll in log_likelihoods):
             # Handle case where all scoring failed
             predicted_class = model_classes[0] # Default to first class or handle differently
             logger.warning(f"All models failed to score sample {i}. Defaulting prediction to {predicted_class}.")
        else:
            predicted_class_index = np.argmax(log_likelihoods)
            predicted_class = model_classes[predicted_class_index]
            
        y_pred.append(predicted_class)

    y_pred = np.array(y_pred)

    end_time = time.time()
    logger.info(f"HMM evaluation finished in {end_time - start_time:.2f} seconds.")

    # Log detailed metrics
    # Using "full" to get the classification report and AUC (if applicable)
    # y_scores is None for HMM in this implementation, metric_results handles this
    logger.info("Calculating and logging detailed evaluation metrics...")
    metrics_report = metric_results(y_true, y_pred, None, ["full"])
    if metrics_report and metrics_report.strip():
        logger.info("--- HMM Evaluation Results ---")
        logger.info(f"\n{metrics_report}")
    else:
        logger.warning("Metrics report was empty for HMM.")

    # HMM log-likelihoods aren't directly probabilities for metrics like AUC
    # Return None for y_scores unless a calibration method is implemented
    return y_true, y_pred, None 