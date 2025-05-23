import joblib
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from typing import Dict, Any, Tuple
from tqdm import tqdm
import time

from utils.logger import get_logger
from utils.testutils import metric_results

logger = get_logger("svm_eval")

def evaluate(dataset: Any, saved_path: Path, model_args: Dict[str, Any] = {}) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates a trained SVM model on the provided dataset object using batch processing.

    Args:
        dataset (ImageDataset): An object containing the feature matrix (X)
                                  and the target vector (y) as attributes.
        saved_path (Path): The path to the saved (trained) SVM model file (.joblib).
        model_args (Dict[str, Any]): Dictionary of arguments (currently unused in evaluate
                                     but kept for consistency).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the true labels (y),
                                                   predicted labels (y_pred), and prediction
                                                   probabilities/scores (y_scores).

    Raises:
        FileNotFoundError: If the model file at saved_path does not exist.
        ValueError: If the loaded model does not support predict_proba or other issues arise.
        TypeError: If the dataset object is missing 'images' or 'labels' attributes.
    """
    start_time = time.time()
    batch_size = model_args.get("batch_size", 128)
    logger.info(f"Starting SVM evaluation with batch size: {batch_size}")

    # Extract data from the dataset object
    try:
        X = dataset.images
        y = dataset.labels
    except AttributeError:
        logger.error("The provided dataset object does not have 'images' or 'labels' attributes.")
        raise TypeError("Dataset object must have 'images' and 'labels' attributes.")

    # Flatten the images if they are not already flat
    if X.ndim > 2:
        original_shape = X.shape
        n_samples = X.shape[0]
        X = X.reshape(n_samples, -1)
        logger.info(f"Flattened test data from {original_shape} to {X.shape}")

    logger.info(f"Evaluation dataset shape: X={X.shape}, y={y.shape}")

    # Ensure target vector is 1D
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
        logger.info("Flattened target vector to 1D")
    elif y.ndim > 1:
        logger.error(f"Target vector y has unexpected shape: {y.shape}")
        raise ValueError(f"Target vector y has unexpected shape: {y.shape}")

    # Check if model file exists
    if not saved_path.is_file():
        logger.error(f"Model file not found at: {saved_path}")
        raise FileNotFoundError(f"Model file not found at: {saved_path}")

    # Load the trained model
    try:
        model: SVC = joblib.load(saved_path)
        logger.info(f"SVM model loaded successfully from {saved_path}")
        
        # Log model details
        n_support = model.n_support_
        logger.info(f"Number of support vectors per class: {n_support}")
        logger.info(f"Total number of support vectors: {sum(n_support)}")
        
    except Exception as e:
        logger.error(f"Failed to load the model from {saved_path}: {e}")
        raise

    n_samples = X.shape[0]
    y_pred_list = []
    y_scores_list = []
    model_supports_proba = hasattr(model, "predict_proba")

    # Make predictions in batches
    logger.info("Starting batch prediction...")
    try:
        for i in tqdm(range(0, n_samples, batch_size), desc="Predicting batches"):
            batch_X = X[i:min(i + batch_size, n_samples)]

            # Predict labels
            batch_y_pred = model.predict(batch_X)
            y_pred_list.append(batch_y_pred)

            # Predict probabilities (if supported)
            if model_supports_proba:
                try:
                    batch_y_scores = model.predict_proba(batch_X)
                    y_scores_list.append(batch_y_scores)
                except Exception as proba_e:
                    logger.error(f"Error predicting probabilities for batch starting at index {i}: {proba_e}")
                    model_supports_proba = False
                    y_scores_list = []  # Clear any partial results
            elif not y_scores_list:  # Only log once
                logger.warning("Model does not support probability predictions.")

        # Concatenate results from batches
        y_pred = np.concatenate(y_pred_list)
        if y_scores_list:
            y_scores = np.concatenate(y_scores_list)
        else:
            logger.warning("Returning zero array for scores as probability prediction was not possible.")
            y_scores = np.zeros((n_samples, len(np.unique(y))))

        logger.info("Batch prediction completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during batch prediction: {e}")
        raise

    end_time = time.time()
    logger.info(f"SVM evaluation finished in {end_time - start_time:.2f} seconds.")

    # Log detailed metrics
    logger.info("Calculating and logging detailed evaluation metrics...")
    metrics_report = metric_results(y, y_pred, y_scores, ["full"])
    if metrics_report and metrics_report.strip():
        logger.info("--- SVM Evaluation Results ---")
        logger.info(f"\n{metrics_report}")
    else:
        logger.warning("Metrics report was empty for SVM.")

    return y, y_pred, y_scores
