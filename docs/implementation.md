# Implementation

## Repository structure
```
ml-pupils
|── docs
|── notebooks
|──src
    │── dataset
    │── models
    │── results/models
    │── utils
|── tests
|── requirements.txt 
```

## Details of `src` directory:
### run.py
Orchestrate the flow of arguments parsing, path definition, data preprocessing, training and evaluating.

### Dataset
- **raw/**: Save downloaded dataset from kaggle. Include 2 subdirectories `Training/` and `Testing/`, each contain 4 sub-sub-directories corresponding to classes of images.
- **processed/**: Save processed data in `.npz` format. Newly created dataset will be processed and save as `train.npz` and `test.npz`.

### Models
Include python files implementing ML models, for example decision_tree.py. Each model must implement a `train` and a `evaluate` function 
```
train(dataloader, save_dir, checkpoint_path)
```
- **dataloader**: tuple of inputs, labels for sklearn models, or torch.utils.data.DataLoader for torch models
- **save_dir**: the directory to save the trained model, handle by run.py
- **checkpoint_path**: used by ANN (torch) models to resume checkpoint before training.

```
evaluate(dataloader, save_path)
```
- **dataloader**: the same as train
- **save_path**: path to the saved model, handle by run.py

### Utils
Include dataset preprocessing, dataloading, and utilities
- **preprocessing.py**: `process_image` handle the load and preprocess of images into numpy.array
- **dataloader.py**: encapsulate data in `ImageDataset`, use `get_dataloader` to fetch the data. When an instance of ImageDataset is created, the images and labels will be processed by `preprocessing.process_image` and save to `dataset/processed` as .npz file  
- **testutils.py**: include function for testing and evaluating on different metrics
- **utils.py**: other utilities

### Results
Store running log and results
- **models/**: store saved models in their corresponding directories.


## Future improvement:
- Using cross validation for training