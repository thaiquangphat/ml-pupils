from .bayes_net import *
from .utils import extract_features_from_imagedataset, build_bayesian_network

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