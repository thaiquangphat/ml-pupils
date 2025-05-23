## [1.0.0] - 23-02-2025
### Removed
- Delete initial `src` structure

### Added
- Data downloading and preprocessing pipeline for `preprocessing.py`
- Implement Dataloader for getting input to models
- Add utilities for getting paths, download kaggle dataset
- Add testing on evaluation metrics
- Implement train and evaluate functions for `decision_tree.py` and `ann.py` models, **no cross-validation**
- Simple LeNet5 models for `ann.py`, adding batch normalization 
- Features for saving processed data and trained model
- Add docs

## [1.0.1] - 25-02-2025
### Modified
- `train` function parameters changed, use **args** for model-specific argument passing
- Update implementation and usage docs

### Added
- Handle argument parsing with configuration file.
- Add default arguments for ann.py

## [1.0.2] 28-02-2025
### Added
- Perform Exploratory Data Analysis (EDA) on current features
- Feature extraction explanation

## [1.0.3] 02-03-2025
### Modified
- Improve feature extraction using GLCM
- Improve Bayesian network structure from EDA
- Added option `bayes` along with `mle`
- Visualization option is simplified

### Removed
- Similarity Redundancy in Naive Bayes and Bayes Network model: remove `naive_bayes.py`
  
## [1.1.0] - 03-03-2025
### Modified: 
- Move dataloader to model scope, run.py only provide ImageDataset object
- Change `get_dataloader` to `get_dataset` to accomodate the above change.
- Move config files to a separate folder `src/config/`
- Modify docs for `train` and `evaluate` implementation

### Added
- Add validation for ann.py checkpointing and early stopping

## [1.1.0] 05-03-2025
### Modified
- Model Introduction and Pipeline Explanation in README.md

## [1.1.1] 09-03-2025
### Added
- Logger for saving training and evaluation process and result

## [1.1.1] 11-03-2025
## Added
- Model's Result Evaluation is added in README.md

## [1.2.0] - 18-03-2025
### Modified:
- Separate train and evaluate function into separate files under their model folder.
- Change run.py accordingly to import train and evaluate function dynamically

### Added:
- Add GA model implementation into pipeline

## [2.0.0] - 22-04-2025
### Added:
- Add SVM model implementation

## [2.1.0] - 24-04-2025
### Added:
- Add HMM implementation

## [2.2.0] - 01-05-2025
### Modified
- Add logger feature for the training and evaluation process of both SVM and HMM models.

## [2.3.0] - 18-05-2025
### Modified
- Fixed minor bugs
- Recheck the models' performance.

### Added
- Add documentation about SVM and HMM on README.