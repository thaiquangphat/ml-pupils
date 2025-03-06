# ML Pupils
Machine learning course @ HCMUT

### Decision Tree
We use `DecisionTreeClassifer` from sklearn.tree with grid search on 5-fold cross validation.
The hyperparameters that need tuning include:
- `criterion`: ['gini', 'entropy']. Gini index one of the feature selection metrics that is used for building binary tree with faster computation, while Entropy is suitable for multi-class classification problem. We not tuning with 'log_loss' as it is more suitable for binary classification, which is not the case.
- `max_features`: [50..10000]. As there are 256x256 features in an input array, the time complexity when choosing the best split is huge. To increase training time while maintain the model performance, we try to tune max_features to different values that is smaller than the original space.
- `min_samples_leaf`: [50,100]. For generalization, we want to optimize the minimum samples at the leaf node, which is equivalent to performing a post pruning.
- Other pruning-related hyperparameters such as `max_depth` is not used as we run on multiple trials and the max_depth in our case is not very deep.

For grid search, we using 'recall_macro' scoring strategy. The reason is that our task is to classify medical image, focusing on the increasing the number of correct prediction for having brain tumor, for which `recall` is most valuable metrics. Between 'recall_macro' and 'recall_micro', we use 'recall_macro' as our datasets split into 4 equal categories of brain tumor.

The result of each running time is logged in results/log and further analysis is conducted in notebooks/result_analysis.

### ANN
The model is based on simple LeNet5 model