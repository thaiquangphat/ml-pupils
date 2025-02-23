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

