import os
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from utils.utils import save_pkl, load_pkl, get_save_name, discretize_data
from features.build_features import segment_and_extract_features
from tqdm import tqdm
from pathlib import Path
from visualize.visualization import visualize_bayesian_network
from utils.logger import get_logger
import time

logger = get_logger("bayes_net")

COLS = ['area', 'perimeter', 'eccentricity', 'solidity', 'contrast', 'homogeneity', 'energy', 'correlation']

DEFAULT_ARGS = {
    "discretize_features": True,
    "bins": 3,
    "estimator": "mle",
    "naive": False,
    "chunk_size": 1000,
    "output_dir": "features/output",
    "visualization": {
        "enabled": True,
        "type": "network",
        "output_file": "results/figures/bayes_net.png",
        "show": False
    }
}

def build_bayesian_network(naive=False):
    """Define BN structure with more tumor features"""
    if naive:
        # Naive Bayes structure: class variable is parent of all features
        model = BayesianNetwork([
            ("tumor_type", "area"),
            ("tumor_type", "perimeter"),
            ("tumor_type", "eccentricity"),
            ("tumor_type", "solidity"),
            ("tumor_type", "contrast"),
            ("tumor_type", "homogeneity"),
            ("tumor_type", "energy"),
            ("tumor_type", "correlation")
        ])
        print("Building Naive Bayes structure...")
    else:
        # Regular Bayesian network structure
        model = BayesianNetwork([
            # tumor_type influences discriminative features directly
            ('tumor_type', 'area'),
            ('tumor_type', 'eccentricity'),
            ('tumor_type', 'homogeneity'),
            ('tumor_type', 'contrast'),
            
            # Feature dependencies based on correlations
            ('area', 'perimeter'),
            ('eccentricity', 'solidity'),
            ('homogeneity', 'energy'),
            ('energy', 'correlation')
        ])
        print("Building regular Bayesian Network structure...")
        print("Nodes:", model.nodes())
        print("Edges:", model.edges())
    return model

def extract_features_from_imagedataset(dataset, output_dir="features/output", split_name="train", chunk_size=1000):
    """
    Extract features directly from an ImageDataset object.
    """
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, f"features_{split_name}.csv")
    if os.path.exists(features_path):
        print(f"Features already extracted for {split_name}. Skipping.")
        return features_path
    
    try:
        if not hasattr(dataset, 'images') or not hasattr(dataset, 'labels'):
            raise ValueError("Dataset object must have 'images' and 'labels' attributes")
        
        images = dataset.images
        labels = dataset.labels
        
        print(f"Processing {len(images)} images from ImageDataset")
        features_list = []
        chunk_counter = 0
        
        # Class mapping
        class_names = ["notumor", "glioma", "meningioma", "pituitary"]
        
        # Process each image with tqdm progress bar
        for idx in tqdm(range(len(images)), desc="Extracting features", unit="img"):
            img = images[idx]
            label_idx = int(labels[idx])
            
            # Map label index to name
            label = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
            
            # Extract features
            is_tumor = (label != "notumor")
            feats = segment_and_extract_features(img, is_tumor)
            feats["label"] = label
            feats["split"] = split_name
            features_list.append(feats)
            
            # Save in chunks to manage memory
            if len(features_list) >= chunk_size or idx == len(images) - 1:
                df_chunk = pd.DataFrame(features_list)
                
                mode = 'w' if chunk_counter == 0 else 'a'
                header = True if chunk_counter == 0 else False
                df_chunk.to_csv(features_path, mode=mode, header=header, index=False)
                
                features_list = []
                chunk_counter += 1
        
        return features_path
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def train(dataset, save_dir, model_args=None):
    """
    Train a Bayesian Network model on the dataset.
    
    Parameters:
    -----------
    dataset : ImageDataset
        Dataset object containing images and labels
    save_dir : Path
        Directory to save the trained model
    model_args : dict
        Model-specific arguments
    """
    start_time = time.time()

    # Merge default args with provided args
    if model_args is None:
        model_args = {}
    args = {**DEFAULT_ARGS, **model_args}
    logger.info(f"Model configuration: {args}")
    
    # Extract features from the dataset
    features_path = extract_features_from_imagedataset(
        dataset, 
        output_dir=args.get("output_dir", "features/output"),
        split_name="train",
        chunk_size=args.get("chunk_size", 1000)
    )
    
    if features_path is None:
        raise ValueError("Feature extraction failed")
    
    # Get naive parameter
    naive = args.get("naive", False)
    model_type = "Naive Bayes" if naive else "Bayesian Network"
    logger.info(f"=== Start {model_type} Training ===")
    
    # Load data
    df = pd.read_csv(features_path)
    train_df = df[df['split'] == 'train']  # Just in case
    logger.info(f"Loaded {len(train_df)} training samples")
    
    feature_cols = COLS
    columns_to_use = feature_cols + ['label']
    model_df = train_df[columns_to_use].copy()
    
    # Rename 'label' to 'tumor_type' to match BN structure
    model_df.rename(columns={'label': 'tumor_type'}, inplace=True)
    
    # Discretize features if needed
    if args["discretize_features"]:
        model_df = discretize_data(model_df, bins=args["bins"], numerical_cols=COLS)
        
    # Build model with the naive parameter
    model = build_bayesian_network(naive=naive)
    
    # Store the discretization parameters with the model
    model.discretization_params = {
        "bins": args["bins"],
        "discretize_features": args["discretize_features"],
        "naive": naive,
        "output_dir": args.get("output_dir", "features/output"),
        "chunk_size": args.get("chunk_size", 1000)
    }
    
    # Estimate parameters
    if args["estimator"] == "mle":
        logger.info("Using Maximum Likelihood Estimator...")
        model.fit(model_df, estimator=MaximumLikelihoodEstimator)
    elif args["estimator"] == "bayes":
        logger.info("Using Bayesian Estimator with BDeu prior...")
        model.fit(model_df, estimator=BayesianEstimator, prior_type="BDeu")
    else:
        error_msg = f"Unknown estimator: {args['estimator']}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Print CPDs
    logger.info("CPDs after training:")
    for cpd in model.get_cpds():
        logger.info(f"\nNode: {cpd.variable}\n{cpd}")
    
    # Save model
    save_path = Path(save_dir) / get_save_name("bayes_net", "pkl")
    save_pkl(model, save_path)
    logger.info(f"Model saved at {save_path}")

    # Log total training time
    total_time = time.time() - start_time
    logger.info(f"=== {model_type} training completed in {total_time:.2f} seconds ===")
    return model

def evaluate(dataset, saved_path, model_args=None):
    """
    Evaluate Bayesian Network model on test data.
    """
    start_time = time.time()
    if model_args is None:
        model_args = {}
    
    # Load the model
    logger.info(f"Loading model from {saved_path}...")
    model = load_pkl(saved_path)
    
    # Get model parameters from the saved model
    output_dir = model.discretization_params.get("output_dir", "features/output")
    chunk_size = model.discretization_params.get("chunk_size", 1000)
    naive = model.discretization_params.get("naive", False)
    model_type = "Naive Bayes" if naive else "Bayesian Network"
    logger.info(f"=== Starting {model_type} Evaluation ===")
    
    # Extract features from the test dataset
    features_path = extract_features_from_imagedataset(
        dataset, 
        output_dir=output_dir,
        split_name="test",
        chunk_size=chunk_size
    )
    
    if features_path is None:
        raise ValueError("Feature extraction failed")
    
    # Load test data
    test_df = pd.read_csv(features_path)
    
    # Initialize inference engine
    inference = VariableElimination(model)
    
    # Get unique labels
    unique_labels = list(test_df['label'].unique())
    y = []
    y_pred = []
    y_scores = []

    # Process each sample
    errors = 0
    for idx, row in test_df.iterrows():
        try:
            y.append(row['label'])
            # Convert continuous features to discrete bins that match model's expectations
            evidence = {}
            for col in COLS:
                if col in row:
                    # Get the feature value
                    value = row[col]
                    cpd = model.get_cpds(col)
                    if cpd and hasattr(cpd, 'state_names') and col in cpd.state_names:
                        # Use existing state names from model as a guide for discretization 
                        states = sorted([int(s) for s in cpd.state_names[col] if s.isdigit()])
                        if states:
                            # Simple quantile-based assignment (very basic)
                            bin_idx = min(int(len(states) * (value / max(test_df[col]))), len(states)-1)
                            discrete_value = str(states[bin_idx])
                        else:
                            discrete_value = '0'  # Fallback
                    else:
                        discrete_value = '0'  # Fallback
                    evidence[col] = discrete_value

            # Now try inference with cleaned evidence
            result = inference.query(["tumor_type"], evidence=evidence)
            probs = [result.values[result.state_names["tumor_type"].index(label)] 
                    if label in result.state_names["tumor_type"] else 0 
                    for label in unique_labels]

            # Normalize if needed
            if sum(probs) > 0:
                probs = [p/sum(probs) for p in probs]
            else:
                probs = [1.0/len(unique_labels)] * len(unique_labels)
                    
            # Get prediction
            pred_idx = np.argmax(probs)
            y_pred.append(unique_labels[pred_idx])
            y_scores.append(probs)
            
        except Exception as e:
            errors += 1
            logger.error(f"Error processing sample {idx}: {str(e)}")
            # Use most common class as fallback
            most_common = test_df['label'].value_counts().idxmax()
            y_pred.append(most_common)
            probs = np.zeros(len(unique_labels))
            probs[unique_labels.index(most_common)] = 1.0
            y_scores.append(probs)
    
    if errors > 0:
        logger.warning(f"Encountered errors in {errors} samples ({errors/len(test_df)*100:.2f}%)")
    
    # Convert scores to numpy array with shape (n_samples, n_classes)
    y_scores = np.array(y_scores)
    return y, y_pred, y_scores

def visualize(saved_path=None, model=None, args=None):
    """
    Visualize Bayesian Network model.
    
    Parameters:
    -----------
    saved_path : str or Path, optional
        Path to the saved model
    model : BayesianNetwork, optional
        Already loaded model instance
    args : dict, optional
        Visualization arguments
    """
    
    print(model)
    print(args)
    if model is None and saved_path:
        from utils.utils import load_pkl
        model = load_pkl(saved_path)
    
    if model is None:
        print("No model available for visualization")
        return
        
    # Get visualization parameters
    if args is None:
        print("No visualization arguments provided")
        args = {}
        
    viz_config = args.get("visualization", {})
    if not isinstance(viz_config, dict):
        viz_config = {}
        
    # Check if visualization is enabled
    if not viz_config.get("enabled", False):
        return
        
    viz_type = viz_config.get("type", "network")
    output_file = viz_config.get("output_file", None)
    show_viz = viz_config.get("show", False)
    
    # Create output directory if needed
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if viz_type == "network":
        print("Visualizing Bayesian Network structure...")
        visualize_bayesian_network(model, output_file=output_file, show=show_viz)
    else:
        print(f"Unknown visualization type: {viz_type}")
    print("Visualization successful.")

# def calculate_probabilities_manually(model, evidence, unique_labels):
#     """
#     Calculate probabilities manually for both Naive and regular Bayesian networks.
#     This is a fallback method when inference engine fails.
#     """
#     probs = []
    
#     # Get the CPD for tumor_type
#     tumor_type_cpd = model.get_cpds('tumor_type')
    
#     # Check if we have a naive structure
#     is_naive = hasattr(model, 'discretization_params') and model.discretization_params.get('naive', False)
    
#     # Compute p(tumor_type | evidence) for each label
#     for label in unique_labels:
#         # Find the label's index in the CPD
#         if label in tumor_type_cpd.state_names['tumor_type']:
#             label_idx = tumor_type_cpd.state_names['tumor_type'].index(label)
            
#             if is_naive:
#                 # For Naive Bayes: P(C) * ∏ P(F|C)
#                 # Start with prior probability P(C)
#                 prob = tumor_type_cpd.values[label_idx]
                
#                 # Multiply by likelihood of evidence P(F|C) for each feature
#                 for feature, value in evidence.items():
#                     if feature in model.nodes():
#                         feature_cpd = model.get_cpds(feature)
                        
#                         # Find the right probability in the CPD
#                         if value in feature_cpd.state_names[feature] and 'tumor_type' in feature_cpd.state_names:
#                             value_idx = feature_cpd.state_names[feature].index(value)
#                             label_idx = feature_cpd.state_names['tumor_type'].index(label)
                            
#                             # Get the right probability 
#                             prob *= feature_cpd.values[value_idx, label_idx]
#             else:
#                 # For regular Bayesian network: use conditional probabilities
#                 # Get the indices for this specific combination of evidence
#                 evidence_indices = {}
                
#                 # For each parent of tumor_type, get its index in the CPD
#                 for parent in model.get_parents('tumor_type'):
#                     if parent in evidence:
#                         parent_cpd = model.get_cpds('tumor_type')
#                         if parent in parent_cpd.state_names and evidence[parent] in parent_cpd.state_names[parent]:
#                             parent_idx = parent_cpd.state_names[parent].index(evidence[parent])
#                             evidence_indices[parent] = parent_idx
                
#                 # Extract the conditional probability from the CPD using these indices
#                 if evidence_indices:
#                     # Create a tuple of indices in the correct order of the CPD's variables
#                     index_list = []
#                     for var in tumor_type_cpd.variables[1:]:  # Skip the first variable (tumor_type)
#                         if var in evidence_indices:
#                             index_list.append(evidence_indices[var])
#                         else:
#                             # If we don't have evidence for a parent, use index 0
#                             index_list.append(0)
                    
#                     # Get the probability from the CPD
#                     if index_list:
#                         prob = tumor_type_cpd.values[label_idx, tuple(index_list)]
#                     else:
#                         prob = tumor_type_cpd.values[label_idx]
#                 else:
#                     # No parent values in evidence, use marginal
#                     prob = tumor_type_cpd.values[label_idx]
            
#             probs.append(prob)
#         else:
#             # Label not in model's states
#             probs.append(0)
    
#     # Normalize probabilities
#     if sum(probs) > 0:
#         probs = [p/sum(probs) for p in probs]
#     else:
#         probs = [1.0/len(unique_labels)] * len(unique_labels)
    
#     return probs
