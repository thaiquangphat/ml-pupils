import argparse
import importlib
import json
import yaml
from pathlib import Path
from utils.dataloader import get_dataset
from utils.utils import get_latest_model_path, kaggle_download
from utils.testutils import metric_results
import os
import sys

sys.path.append(os.path.abspath(".")) 

os.environ['PYTHONHASHSEED'] = '42'
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
    parser.add_argument("--save_data_dir", type=str, default="dataset/processed", help="Path to saved data.")
    parser.add_argument("--saved_path", type=str, default=None, help="Path to saved ML model.")
    parser.add_argument("--model_args",type=json.loads, default={}, help="Model arguments")
    parser.add_argument("--metrics", nargs="+", type=str, default="full", 
                        help="""List of eval metrics. 
                        Currently support [f1_score, precision_score, recall_score, accuracy_score, auc_score].
                        Use "full" to print a full classification report with auc_score
                        """)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    
    # load config file and update args with config
    if args.config:
        config = load_config(Path("config") /args.config)
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
            

    # Dynamically load the model module
    try:
        train_module = importlib.import_module(f"models.{args.model}.train")
        eval_module = importlib.import_module(f"models.{args.model}.eval")
    except ModuleNotFoundError:
        raise ValueError(f"Model '{args.model}.train|eval' not found. Ensure there is a corresponding file in models/.")

    # Dynamically load train and evaluate functions
    train_func = getattr(train_module, "train", None)
    eval_func = getattr(eval_module, "evaluate", None)

    if not train_func or not eval_func:
        raise ValueError(f"Model '{args.model}' must define 'train' and 'evaluate' functions.")

    # Define model save path
    save_dir = Path(f"results/models/{args.model}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training
    if args.train:
        save_file = save_data_dir / f"train.npz"
        dataset = get_dataset(train_dir, save_path=save_file)
        train_func(dataset, save_dir, args.model_args)

    # Evaluation
    if args.eval:
        save_file = save_data_dir / f"test.npz"
        saved_path = args.saved_path if args.saved_path else get_latest_model_path(save_dir)
        dataset = get_dataset(test_dir, save_path=save_file)
        y, y_preds, y_scores = eval_func(dataset, saved_path, args.model_args)
        print(metric_results(y, y_preds, y_scores, args.metrics))

    print("Execution completed.")
