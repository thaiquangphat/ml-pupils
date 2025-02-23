import argparse
import importlib
import torch
import os
from pathlib import Path
from utils.dataloader import get_dataloader
from utils.utils import get_latest_model_path, kaggle_download
from utils.testutils import metric_results

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run ML models on image dataset.")
    parser.add_argument("--model", type=str, required=True, help="Choose the model to run (e.g., decision_tree, ann).")
    parser.add_argument("--train", action="store_true", help="Train the selected model.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the selected model.")
    parser.add_argument("--dataset", type=str, default="masoudnickparvar/brain-tumor-mri-dataset", help="Kaggle dataset")
    parser.add_argument("--data_dir", type=str, default="dataset/raw", help="Path to dataset.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for ANN.")
    parser.add_argument("--save_data_dir", type=str, default="dataset/processed", help="Path to saved data.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specify checkpoint to load a model.")
    parser.add_argument("--metrics", nargs="+", type=str, default="full", 
                        help="""List of eval metrics. 
                        Currently support [f1_score, precision_score, recall_score, accuracy_score, auc_score].
                        Use "full" to print a full classification report with auc_score
                        """)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Ensure dataset exists
    data_dir = Path(args.data_dir)
    save_data_dir = Path(args.save_data_dir)  
    save_data_dir.mkdir(parents=True, exist_ok=True) 

    if not data_dir.exists():
        kaggle_download(args.dataset, data_dir)

    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"

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

    # Define model save path
    save_dir = Path(f"results/models/{args.model}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # If checkpoint is specified, override save path
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = save_dir / args.checkpoint

    # Training
    # TODO: either split for different types of model (sklearn/torch)
    #       or implement a more comprehensive train/eval func api 
    
    if args.train:
        save_file = save_data_dir / f"train.npz"
        data_loader = get_dataloader(train_dir, save_path=save_file, batch_size=args.batch_size, for_torch="ann" in args.model)
        train_func(data_loader, save_dir, checkpoint_path)

    # Evaluation
    if args.eval:
        save_file = save_data_dir / f"test.npz"
        if not checkpoint_path:
            checkpoint_path = get_latest_model_path(save_dir)
        data_loader = get_dataloader(test_dir, save_path=save_file, batch_size=args.batch_size, for_torch="ann" in args.model)
        y, y_preds, y_scores = eval_func(data_loader, checkpoint_path)
        print(metric_results(y, y_preds, y_scores, args.metrics))

    print("Execution completed.")
