from .bayes_net import *
from .utils import extract_features_from_imagedataset, build_bayesian_network


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