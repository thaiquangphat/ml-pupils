# Implementation

## Repository structure
```
ml-pupils
|── docs
|── notebooks
|──src
    |── config/
    │── dataset/
    │── models/
    │── results/models/
    │── utils/
|── tests
|── requirements.txt 
```

## Details of `src` directory:
### run.py
Orchestrate the flow of arguments parsing, path definition, data preprocessing, training and evaluating.

### Config
.yaml config files should be stored here.

### Dataset
- **raw/**: Save downloaded dataset from kaggle. Include 2 subdirectories `Training/` and `Testing/`, each contain 4 sub-sub-directories corresponding to classes of images.
- **processed/**: Save processed data in `.npz` format. Newly created dataset will be processed and save as `train.npz` and `test.npz`.

### Models
Include python files implementing ML models, for example decision_tree.py. Each model must implement a `train` and a `evaluate` function 
#### Train
```
train(dataset, save_dir, **args**)
```
- **dataset**: ImageDataset() inherit from torch.utils.data.Dataset. In the new modification, for sklearn model, use `dataset.images` and `dataset.labels` to get numpy.array() data; pytorch models will use this to initialize Dataloader() for the own needs.
- **save_dir**: the directory to save the trained model, handle by run.py
- **args**: necessary arguments for a particular model, should be specified in config file under "model_args"

##### Guide: 
- To implement `train` for a new model, **dataset** and **save_dir** input format must follow the description above. 
- Other model-specific arguments can be passed through **args**, use global DEFAULT_ARGS (dict) to ensure the existent of arguments (see ann.py for example).
- After training finished, the trained model must be saved using `utils.utils.save_pkl` or `torch.save` for later use.

#### Evaluate
```
evaluate(dataset, save_path,args)
```
- **dataset**: the same as train
- **saved_path**: path to the saved model, handle by run.py
- **args**: (dict) model-specific arguments

##### Guide:
- To implement `eval` for a new model, **dataset** and **saved_path** input format must follow the description above. 
- Must use **saved_path** to load model by `utils.utils.load_pkl` or `torch.load` (remember to handle the None case).
- Any model-specific evaluation metrics should be implemented in `utils.testutils` and import before use.

### Utils
Include dataset preprocessing, dataloading, and utilities
- **preprocessing.py**: `process_image` handle the load and preprocess of images into numpy.array
- **dataloader.py**: encapsulate data in `ImageDataset`, use `get_dataset` to get the ImageDataset object. When an instance of ImageDataset is created, the images and labels will be processed by `preprocessing.process_image` and save to `dataset/processed` as .npz file  
- **testutils.py**: include function for testing and evaluating on different metrics
- **utils.py**: other utilities

### Results
Store running log and results
- **models/**: store saved models in their corresponding directories.


## Future improvement:
