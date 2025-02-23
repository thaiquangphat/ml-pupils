# USAGE

## Quick start
Clone the repo, go the src/ directory, run with command:
```
python -m run --model ann [options]
```
Options include:
- `--model`[required]: One of the models defined in src/models/. Currently support ["decision_tree", "ann"]
- `--train`: If specified, the model will be trained on Training dataset
- `--eval`: If specified, the model will be tested on Testing dataset. Note: at least one of `--train` and `--eval` must be enabled
- `--dataset`[optional]: Kaggle dataset name; currently working on [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/) specifically
- `--data_dir`[optional]: Directory path to raw dataset. Default: `dataset/raw/` -> if directory doesn't exist, the dataset will be downloaded.
- `--save_data_dir`[optional]: Directory path to saved processed data files.Default: `dataset/processed/`
- `--checkpoint`[optional]: Checkpoint file path for loading saved model. Can be a checkpoint for ANN models or fully trained scikit-learn models. If not specified, the model will be trained from scratch.
- `--metrics`[optional]: List of test metrics on the test set. Currently support ["full", "accuracy_score", "f1_score", "precision_score", "recall_score", "auc_score"]. Default: ["full"] -> print classification report
- `--batch_size`[optional]: For batch dataloading of ANN model. Default: 64

## Example usage:
```
python -m run py --model ann --train --eval --batch_size 32 --metrics accuracy_score auc_score
```
This will train the model ANN from scratch using a batch size of 32 and evaluate on accuracy_score and auc_score.