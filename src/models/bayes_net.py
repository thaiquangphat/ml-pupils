import os
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from utils.utils import save_pkl, load_pkl, get_save_name

DEFAULT_ARGS = {
    "discretize_features": True,
    "bins": 5,
    "estimator": "mle"  # 'mle' or 'bayes'
}

def build_bayesian_network(naive=False):
    """Define BN structure with more tumor features"""
    if naive:
        # Naive Bayes structure: class variable is parent of all features
        model = BayesianNetwork([
            ("tumor_type", "tumor_present"),
            ("tumor_type", "area"),
            ("tumor_type", "perimeter"),
            ("tumor_type", "mean_intensity"),
            ("tumor_type", "eccentricity"),
            ("tumor_type", "solidity"),
            ("tumor_type", "contrast"),
            ("tumor_type", "homogeneity")
        ])
        print("Building Naive Bayes structure...")
    else:
        # Regular Bayesian network structure
        model = BayesianNetwork([
            ("tumor_present", "tumor_type"),
            ("area", "tumor_type"),
            ("perimeter", "tumor_type"),
            ("mean_intensity", "tumor_type"),
            ("eccentricity", "tumor_type"),
            ("solidity", "tumor_type"),
            ("contrast", "tumor_type"),
            ("homogeneity", "tumor_type")
        ])
        print("Building regular Bayesian Network structure...")
    return model

def _discretize_data(df, bins=5):
    """Discretize continuous features for Bayesian Network."""
    numerical_cols = ['area', 'perimeter', 'mean_intensity', 'eccentricity', 
                      'solidity', 'contrast', 'homogeneity']
    df_disc = df.copy()
    
    for col in numerical_cols:
        if col in df.columns:
            # Handle zero or very small values that might cause issues
            if df[col].min() == df[col].max():
                df_disc[col] = '0'
            else:
                df_disc[col] = pd.qcut(df[col], q=bins, labels=False, duplicates='drop')
    
    # Ensure all columns are strings for pgmpy
    for col in df_disc.columns:
        df_disc[col] = df_disc[col].astype(str)
        
    return df_disc

def train(features_path, save_dir, args):
    """
    Train a Bayesian Network model on extracted features.
    
    Parameters:
    -----------
    features_path : str or Path
        Path to CSV file with extracted features
    save_dir : Path
        Directory to save the trained model
    args : dict
        Model-specific arguments
    """
    args = {**DEFAULT_ARGS, **args}
    naive = args.get("naive", False)  # Get naive parameter with default False

    # Load data
    df = pd.read_csv(features_path)
    train_df = df[df['split'] == 'train']
    
    feature_cols = ['tumor_present', 'area', 'perimeter', 'mean_intensity', 
                  'eccentricity', 'solidity', 'contrast', 'homogeneity']
    columns_to_use = feature_cols + ['label']
    model_df = train_df[columns_to_use].copy()
    
    # Rename 'label' to 'tumor_type' to match BN structure
    model_df.rename(columns={'label': 'tumor_type'}, inplace=True)
    
    # Discretize features if needed
    if args["discretize_features"]:
        model_df = _discretize_data(model_df, bins=args["bins"])
        
    # Build and train model
    model = build_bayesian_network(naive=naive)
    
    # IMPORTANT: Store the discretization parameters with the model
    model.discretization_params = {
        "bins": args["bins"],
        "discretize_features": args["discretize_features"],
        "naive": naive
    }
    
    # Use appropriate estimator
    if args["estimator"] == "mle":
        model.fit(model_df, estimator=MaximumLikelihoodEstimator)
    else:
        model.fit(model_df, estimator=BayesianEstimator, prior_type="BDeu", pseudo_counts=1)
    
    print("CPTs after training:")
    for node in model.nodes():
        print(f"Node: {node}")
        print(model.get_cpds(node))
        # Verify that probabilities sum to 1
        if node == "tumor_present":
            cpd = model.get_cpds(node)
            print("Values sum to:", cpd.values.sum())
            print("Values shape:", cpd.values.shape)
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / get_save_name("bayes_net", "pkl")
    save_pkl(model, save_path)
    print(f"Model saved at {save_path}")
    
    return model

def evaluate(features_path, saved_path):
    """
    Evaluate Bayesian Network model on test data.
    
    Parameters:
    -----------
    features_path, str or Path: Path to CSV file with extracted features
    saved_path, str or Path: Path to saved model file
        
    Returns:
    --------
    y_true : list: True labels
    y_pred : list: Predicted labels
    y_scores : list: Probability scores for each class
    """
    if not saved_path or not os.path.exists(saved_path):
        raise FileNotFoundError("Model not found. Please train first.")
    
    model = load_pkl(saved_path)
    test_df = pd.read_csv(features_path)

    # Apply the same discretization to test data
    if hasattr(model, 'discretization_params') and model.discretization_params['discretize_features']:
        # Rename 'label' to 'tumor_type' for consistency with model
        test_df_copy = test_df.copy()
        test_df_copy.rename(columns={'label': 'tumor_type'}, inplace=True)
        test_df_copy = _discretize_data(test_df_copy, bins=model.discretization_params['bins'])
        # Rename back
        test_df_copy.rename(columns={'tumor_type': 'label'}, inplace=True)
        test_df = test_df_copy
    
    
    # Get the set of unique labels
    unique_labels = sorted(list(test_df['label'].unique()))
    
    print(f"Evaluating on {len(test_df)} test samples with {len(unique_labels)} classes")
    
    # Get ground truth labels
    y_true = test_df['label'].tolist()
    
    # Make predictions
    y_pred = []
    y_scores = []
    
    # Initialize the inference engine
    try:
        inference = VariableElimination(model)
    except Exception as e:
        print(f"Error initializing inference engine: {e}")
        # Fallback to most common class
        most_common = test_df['label'].value_counts().idxmax()
        y_pred = [most_common] * len(test_df)
        y_scores = [[1.0 if l == most_common else 0.0 for l in unique_labels]] * len(test_df)
        return y_true, y_pred, y_scores
    
    # Process each sample
    for idx, row in test_df.iterrows():
        try:
            # Prepare evidence - convert all feature values to strings
            evidence = {}
            for col in ['tumor_present', 'area', 'perimeter', 'mean_intensity', 
                      'eccentricity', 'solidity', 'contrast', 'homogeneity']:
                if col in row:
                    evidence[col] = str(row[col])
            
            # Try using the inference engine, fallback to manual calculation if it fails
            try:
                # For regular BN, try with inference engine first
                result = inference.query(["tumor_type"], evidence=evidence)
                
                # Extract probabilities for each label
                probs = np.zeros(len(unique_labels))
                
                # Map 'label' to 'tumor_type' in state_names
                for i, label in enumerate(unique_labels):
                    if label in result.state_names["tumor_type"]:
                        idx = result.state_names["tumor_type"].index(label)
                        probs[i] = result.values[idx]
                        
                # Normalize if needed
                if np.sum(probs) > 0:
                    probs = probs / np.sum(probs)
                    
            except Exception as e:
                print(f"Inference failed for sample {idx}, using manual calculation: {e}")
                probs = calculate_probabilities_manually(model, evidence, unique_labels)
                
            # Get prediction
            pred_idx = np.argmax(probs)
            y_pred.append(unique_labels[pred_idx])
            y_scores.append(probs)
            
        except Exception as e:
            print(f"Error predicting sample {idx}: {e}")
            # Use most common class as fallback
            most_common = test_df['label'].value_counts().idxmax()
            y_pred.append(most_common)
            probs = np.zeros(len(unique_labels))
            probs[unique_labels.index(most_common)] = 1.0
            y_scores.append(probs)
    
    # Final consistency check
    if len(y_pred) == 0:
        print("Warning: No predictions made. Using default values.")
        most_common = test_df['label'].value_counts().idxmax()
        y_pred = [most_common] * len(y_true)
        y_scores = [[1.0 if l == most_common else 0.0 for l in unique_labels]] * len(y_true)
    
    print(f"Successfully generated {len(y_pred)} predictions")
    
    return y_true, y_pred, y_scores

# Fallback function
def calculate_probabilities_manually(model, evidence, unique_labels):
    """
    Calculate probabilities manually using Bayes' rule.
    
    Parameters:
    -----------
    model : BayesianNetwork
        Trained Bayesian Network model
    evidence : dict
        Dictionary of evidence variables
    unique_labels : list
        List of unique label values
        
    Returns:
    --------
    probs : list
        List of probabilities for each label
    """
    probs = []
    
    # Get the CPD for tumor_type
    tumor_type_cpd = model.get_cpds('tumor_type')
    
    # Check if we have a naive structure (class -> features) or regular (features -> class)
    is_naive = hasattr(model, 'discretization_params') and model.discretization_params.get('naive', False)
    
    # Compute p(tumor_type | evidence) for each label
    for label in unique_labels:
        # Find the label's index in the CPD
        if label in tumor_type_cpd.state_names['tumor_type']:
            label_idx = tumor_type_cpd.state_names['tumor_type'].index(label)
            
            if is_naive:
                # For Naive Bayes: P(C) * ∏ P(F|C)
                # Start with prior probability P(C)
                prob = tumor_type_cpd.values[label_idx]
                
                # Multiply by likelihood of evidence P(F|C) for each feature
                for feature, value in evidence.items():
                    if feature in model.nodes():
                        feature_cpd = model.get_cpds(feature)
                        
                        # Find the right probability in the CPD
                        if value in feature_cpd.state_names[feature] and label in feature_cpd.state_names['tumor_type']:
                            value_idx = feature_cpd.state_names[feature].index(value)
                            label_idx = feature_cpd.state_names['tumor_type'].index(label)
                            prob *= feature_cpd.values[value_idx, label_idx]
            else:
                # For regular Bayesian network: P(C|F1,F2,...) directly from CPD
                # Get the indices for this specific combination of evidence
                evidence_indices = {}
                
                # For each parent of tumor_type, get its index in the CPD
                for parent in model.get_parents('tumor_type'):
                    if parent in evidence:
                        parent_cpd = model.get_cpds('tumor_type')
                        if parent in parent_cpd.variables and evidence[parent] in parent_cpd.state_names[parent]:
                            parent_idx = parent_cpd.state_names[parent].index(evidence[parent])
                            evidence_indices[parent] = parent_idx
                
                # Extract the conditional probability from the CPD using these indices
                if evidence_indices:
                    # Create a tuple of indices in the correct order of the CPD's variables
                    index_list = []
                    for var in tumor_type_cpd.variables[1:]:  # Skip the first variable (tumor_type)
                        if var in evidence_indices:
                            index_list.append(evidence_indices[var])
                        else:
                            # If we don't have evidence for a parent, use index 0
                            index_list.append(0)
                    
                    # Get the probability from the CPD
                    prob = tumor_type_cpd.values[label_idx, tuple(index_list)]
                else:
                    # No parent values in evidence, use marginal
                    prob = tumor_type_cpd.values[label_idx]
            
            probs.append(prob)
        else:
            # Label not in model's states
            probs.append(0)
    
    # Normalize probabilities
    if sum(probs) > 0:
        probs = [p/sum(probs) for p in probs]
    else:
        probs = [1.0/len(unique_labels)] * len(unique_labels)
    
    return probs