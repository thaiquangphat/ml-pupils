## [1.2.1] 05-03-2025
### Modified
- Model Introduction and Pipeline Explanation in README.md

## [1.2] 02-03-2025
### Modified
- Improve feature extraction using GLCM
- Improve Bayesian network structure from EDA
- Added option `bayes` along with `mle`
- Visualization option is simplified

## [1.1] 28-02-2025
### Added
- Perform Exploratory Data Analysis (EDA) on current features
- Feature extraction explanation

### Removed
- Similarity Redundancy in Naive Bayes and Bayes Network model: remove `naive_bayes.py`

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

## [1.0.0] - 25-02-2025
### Modified
- `train` function parameters changed, use **args** for model-specific argument passing
- Update implementation and usage docs

### Added
- Handle argument parsing with configuration file.
- Add default arguments for ann.py
