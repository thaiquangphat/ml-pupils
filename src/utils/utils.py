from datetime import datetime
import pickle
import os
from pathlib import Path
from itertools import chain

def get_latest_model_path(directory):
    """Returns the latest model file from a directory."""
    # sort the files by created time
    model_files = sorted(
        chain(
            Path(directory).glob("*.pkl"),  
            Path(directory).glob("*.pth")   
        ),
        key=os.path.getmtime,  
        reverse=True           
    )
    return model_files[0] if model_files else None

def get_save_name(model_name, extension):
    """
    Return the filename for saved model
    Example usage:
        get_save_name("decision_tree", "pkl") -> return pickle file
        get_save_name("ann", "pt") -> return torch.save file
    """
    timestamp = datetime.now().strftime("%d%m%Y_%H-%M-%S")
    return f"{model_name}_{timestamp}.{extension}"

def save_pkl(model, save_path):
    """Save model to using pickle"""
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

def load_pkl(save_path):
    """Load model using pickle"""
    with open(save_path, "rb") as f:
        model = pickle.load(f)
    return model

def kaggle_download(dataset_name, save_dir):
    """
    Downloads a dataset from Kaggle using the Kaggle CLI.
    
    Parameters:
        dataset_name (str): The dataset identifier on Kaggle (e.g., "dataset-owner/dataset-name").
        save_path (str): The directory where the dataset should be saved.
        
    Requirements:
        - The `kaggle` CLI must be installed (`pip install kaggle`).
        - You must authenticate with Kaggle by placing `kaggle.json` in `~/.kaggle/` or `C:/Users/<YourUser>/.kaggle/`.
    
    Example:
        kaggle_download("zynicide/wine-reviews", save_path="data")
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct the Kaggle CLI command
    command = f"kaggle datasets download -d {dataset_name} -p {save_dir} --unzip"

    # Execute the command
    os.system(command)

    print(f"Dataset '{dataset_name}' downloaded and saved to '{save_dir}'.")


    