# USAGE

## Quick start
Clone the repo, go the src/ directory, run with command:
```
python -m run --model ann [options]
```
Options include:
- `--config`: Path to config .yaml file.
- `--model`: One of the models defined in src/models/. Currently support ["decision_tree", "ann"]
- `--train`: If specified, the model will be trained on Training dataset
- `--eval`: If specified, the model will be tested on Testing dataset. Note: at least one of `--train` and `--eval` must be enabled
- `--dataset`: Kaggle dataset name; currently working on [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/) specifically
- `--data_dir`: Directory path to raw dataset. Default: `dataset/raw/` -> if directory doesn't exist, the dataset will be downloaded.
- `--save_data_dir`: Directory path to saved processed data files.Default: `dataset/processed/`
- `--saved_path`: Path to saved model. Can be a checkpoint for ANN models or fully trained scikit-learn models. If not specified, the model will be trained from scratch.
- `--metrics`: List of test metrics on the test set. Currently support ["full", "accuracy_score", "f1_score", "precision_score", "recall_score", "auc_score"]. Default: ["full"] -> print classification report
- `--batch_size`: For batch dataloading of ANN model. Default: 64

## Example usage:
```
python -m run --model ann --train --eval --batch_size 32 --metrics accuracy_score auc_score
```
This will train the model ANN from scratch using a batch size of 32 and evaluate on accuracy_score and auc_score.
OR
```
python -m run --config ./config.yaml --train --eval
```
in `src/config.yaml`
```
model: "ann"
batch_size: 32

model_args:
    epochs: 100
    log_step: 10

metrics: ["accuracy_score", "auc_score"]
```
This will train the ANN model on 100 epochs with logging every 10 epochs.

## Bayesian Network Update
- Dataloader: extracted features from preprocessed npz files, save to directory feature_output (must have the npz files first, so run other models before this)
- New visualization feature for network and feature correlation
- Option `naive`: substitute bayesian network to naive bayes (a special type of bayesian network)
- Challenge: feature extraction is weak thus the model has low accuracy.
- Further improvement: allow more estimator options and improve feature extraction
- Example usage: 
```
python -m run --model bayes_net --naive --train --eval accuracy_score --chunk_size 1000 --visualize --viz_type network --viz_output results/feature_correlations.png --show_viz
```
Explanation: Train the naive bayes model by extracting features from train.npz with chunk size 1000, then evaluate the model by extracting features from test.npz with chunk size 1000 using accuracy_score as metric. Then visualize the naive bayes network, save output to a png file, and show it.

- Should also work with config.yaml file