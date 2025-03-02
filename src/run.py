import argparse
import importlib
import json
import os
import yaml
from pathlib import Path
from utils.dataloader import get_dataloader
from utils.utils import get_latest_model_path, kaggle_download
from utils.testutils import metric_results
from utils.visualization import visualize_bayesian_network

# Feature extraction for (Naive) Bayes Network
from utils.feature_extraction import process_dataset

# function to load config.json file
def load_config(filepath):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
    
# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run ML models on image dataset.")
    parser.add_argument("--config",type=str, help="Path to config JSON file")
    parser.add_argument("--model", type=str, help="Choose the model to run (e.g., decision_tree, ann).")
    parser.add_argument("--train", action="store_true", help="Train the selected model.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the selected model.")
    parser.add_argument("--dataset", type=str, default="masoudnickparvar/brain-tumor-mri-dataset", help="Kaggle dataset")
    parser.add_argument("--data_dir", type=str, default="dataset/raw", help="Path to dataset.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for ANN.")
    parser.add_argument("--save_data_dir", type=str, default="dataset/processed", help="Path to saved data.")
    parser.add_argument("--saved_path", type=str, default=None, help="Path to saved ML model.")
    parser.add_argument("--metrics", nargs="+", type=str, default="full", 
                        help="""List of eval metrics. 
                        Currently support [f1_score, precision_score, recall_score, accuracy_score, auc_score].
                        Use "full" to print a full classification report with auc_score
                        """)
    parser.add_argument("--features_path", type=str, default="features_train.csv",
                      help="Path to CSV file with extracted features (for feature-based models)")
    parser.add_argument("--chunk_size", type=int, default=1000,
                      help="Number of images to process per chunk for large NPZ files")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--viz_output", type=str, default=None, 
                        help="Path to save visualization output")
    parser.add_argument("--show_viz", action="store_true", help="Display visualizations")
    parser.add_argument("--viz_type", type=str, default="network", 
                    choices=["network", "features"],
                    help="Type of visualization to generate")
    parser.add_argument("--naive", action="store_true", 
                      help="Use Naive Bayes variant for bayes_net model")
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    
    # load config file and update args with config
    if args.config:
        config = load_config(args.config)
        args.__dict__.update(config) 
    
    # A model name is required, either pass by cli for mention in config file
    if not args.model:
        raise ValueError("No configuration for model found")
    
    # Ensure dataset exists
    data_dir = Path(args.data_dir)
    save_data_dir = Path(args.save_data_dir)  
    save_data_dir.mkdir(parents=True, exist_ok=True) 

    # download directly from kaggle if not exist
    if not data_dir.exists():
        kaggle_download(args.dataset, data_dir)
    
    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"

    # Define model save path
    save_dir = Path(f"results/models/{args.model}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
   # Model is required for training/evaluation
    if not args.model:
        raise ValueError("No configuration for model found")
        
    # Dynamically load the model module
    try:
        model_module = importlib.import_module(f"models.{args.model}")
    except ModuleNotFoundError:
        raise ValueError(f"Model '{args.model}' not found. Ensure there is a corresponding file in models/.")

    # Dynamically load train and evaluate functions
    train_func = getattr(model_module, "train", None)
    eval_func = getattr(model_module, "evaluate", None)

    if not train_func or not eval_func:
        raise ValueError(f"Model '{args.model}' must define 'train' and 'evaluate' functions.")
        
    # Training
    if args.train:
        save_file = save_data_dir / f"train.npz"
        if args.model == "bayes_net":
            # Use feature-based training
            full_path = f"feature_output/{args.features_path}"
            if not os.path.exists(full_path):
                print(f"Features file not found. Extracting features from {save_file} file...")
                process_dataset(dataset=save_file, output=full_path, chunk_size=args.chunk_size)
            # Get model_args safely
            model_args = {} if not hasattr(args, 'model_args') else args.model_args
            train_func(full_path, save_dir, model_args)
        else:
            save_file = save_data_dir / f"train.npz"
            data_loader = get_dataloader(train_dir, save_path=save_file, batch_size=args.batch_size, for_torch="ann" in args.model)
            model_args = {} if not hasattr(args, 'model_args') else args.model_args
            train_func(data_loader, save_dir, model_args)

    # Evaluation
    if args.eval:
        saved_path = args.saved_path if args.saved_path else get_latest_model_path(save_dir)
        save_file = save_data_dir / f"test.npz"
        if args.model == "bayes_net":
            args.features_path = "features_test.csv"
            # Use feature-based evaluation
            full_path = f"feature_output/{args.features_path}"
            if not os.path.exists(full_path):
                print(f"Features file not found. Extracting features from {save_file} file...")
                process_dataset(dataset=save_file,output=full_path, chunk_size=args.chunk_size)
                print(f"Feature extracted to {full_path}.")
            y, y_preds, y_scores = eval_func(full_path, saved_path)
        else:
            data_loader = get_dataloader(test_dir, save_path=save_file, batch_size=args.batch_size, for_torch="ann" in args.model)
            y, y_preds, y_scores = eval_func(data_loader, saved_path)
        
        print(metric_results(y, y_preds, y_scores, args.metrics))

    # Visualization
    if args.visualize:
        full_path = f"feature_output/{args.features_path}"
        if args.model:
            if args.viz_type == "network":
                if args.model != "bayes_net":
                    print("Network visualization is only available for bayes_net model")
                else:
                    # Load the model if not already loaded
                    saved_path = args.saved_path if args.saved_path else get_latest_model_path(save_dir)
                    
                    if not saved_path:
                        print(f"No saved model found. Please train the model first.")
                    else:
                        from utils.utils import load_pkl
                        model = load_pkl(saved_path)
                        # Visualize the network
                        visualize_bayesian_network(model, args.viz_output, show=args.show_viz)
            
            elif args.viz_type == "features":
                # Visualize feature correlations
                from utils.visualization import visualize_feature_relations
                visualize_feature_relations(features_path=full_path, 
                                        output_file=args.viz_output, 
                                        show=args.show_viz)


    print("Execution completed.")