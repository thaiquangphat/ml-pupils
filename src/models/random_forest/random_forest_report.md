# Random Forest Report

## Overview
This report summarizes the implementation and evaluation of the Random Forest algorithm for image classification, as presented in the `random_forest.ipynb` notebook. The project includes two approaches:
- Using the `RandomForestClassifier` from scikit-learn (library-based)
- A custom re-implementation of Random Forest from scratch

Both approaches use features extracted from images (via a deep learning model) and are evaluated on the same dataset.

---

## 1. Random Forest Using scikit-learn

### Implementation Summary
- **Feature Extraction:** Features are extracted from images using a pre-trained VGG19 model and further processed with PCA for dimensionality reduction.
- **Preprocessing:**
  - Features are scaled using `StandardScaler`.
  - PCA is applied to reduce dimensionality (up to 64 components).
- **Model:**
  - `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)` from scikit-learn is used.
  - The model is trained on the processed features.

```python
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
train_features_pca = pca.fit_transform(train_features_scaled)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(train_features_pca, train_labels)
```

### Results
- **Accuracy:** 0.87
- **Classification Report:**

```
              precision    recall  f1-score   support

      glioma       0.90      0.71      0.80       300
  meningioma       0.71      0.82      0.76       306
     notumor       0.95      0.98      0.96       405
   pituitary       0.91      0.92      0.92       300

    accuracy                           0.87      1311
   macro avg       0.87      0.86      0.86      1311
weighted avg       0.87      0.87      0.87      1311
```

---

## 2. Custom Random Forest Implementation

### Implementation Summary
- **Decision Tree:**
  - Custom `DecisionTree` class supports max depth, min samples split, and random feature selection for splits.
  - Uses entropy and information gain for splitting.
- **Random Forest:**
  - Custom `RandomForest` class builds multiple `DecisionTree` instances on bootstrap samples.
  - Each tree uses a random subset of features at each split (sqrt of total features).
  - Final prediction is by majority vote across all trees.

### Mathematical Formulation

#### 1. Decision Tree Components

##### Entropy Calculation
For a set of samples $S$ with $C$ classes, the entropy is calculated as:
$$
H(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)
$$
where $p_i$ is the proportion of samples belonging to class $i$.

##### Information Gain
For a split at threshold $t$ on feature $f$:
$$
IG(S, f, t) = H(S) - \left(\frac{|S_L|}{|S|}H(S_L) + \frac{|S_R|}{|S|}H(S_R)\right)
$$
where:
- $S_L$ is the set of samples where $f < t$
- $S_R$ is the set of samples where $f \geq t$
- $|S|$ is the total number of samples
- $|S_L|$ and $|S_R|$ are the number of samples in left and right splits

##### Split Selection
At each node, the algorithm:
1. Randomly selects $\sqrt{n}$ features from $n$ total features
2. For each selected feature $f$:
   - Evaluates all possible thresholds $t$
   - Computes information gain $IG(S, f, t)$
3. Chooses the split with maximum information gain:
$$
(f^*, t^*) = \arg\max_{f,t} IG(S, f, t)
$$

#### 2. Random Forest Construction

##### Bootstrap Sampling
For each tree $t$ in the forest:
1. Sample $n$ instances with replacement from the training set
2. Create a bootstrap sample $B_t$

##### Tree Building
For each tree:
1. Start with root node containing $B_t$
2. At each node:
   - If depth $\geq$ max_depth or samples $<$ min_samples_split:
     - Create leaf node with majority class
   - Else:
     - Find best split using information gain
     - Create left and right child nodes
     - Recursively build subtrees

##### Feature Selection
At each split:
$$
n_{features} = \lfloor\sqrt{n_{total\_features}}\rfloor
$$
where $n_{total\_features}$ is the total number of features.

#### 3. Prediction Process

##### Individual Tree Prediction
For a sample $x$:
1. Start at root node
2. At each node:
   - If leaf node: return class label
   - Else: move to left/right child based on split condition
$$
f_t(x) = \text{leaf\_label}
$$

##### Forest Prediction
Final prediction is majority vote across all trees:
$$
F(x) = \arg\max_{c \in C} \sum_{t=1}^{T} \mathbb{I}(f_t(x) = c)
$$
where:
- $T$ is the number of trees
- $C$ is the set of possible classes
- $\mathbb{I}$ is the indicator function

### Results
- **Accuracy:** 0.85
- **Classification Report:**
```
              precision    recall  f1-score   support

      glioma       0.89      0.70      0.78       300
  meningioma       0.69      0.75      0.72       306
     notumor       0.92      0.97      0.94       405
   pituitary       0.89      0.92      0.91       300

    accuracy                           0.85      1311
   macro avg       0.85      0.84      0.84      1311
weighted avg       0.85      0.85      0.85      1311
```

---

## 3. Comparison and Discussion

| Aspect                | scikit-learn RandomForest | Custom RandomForest |
|-----------------------|--------------------------|--------------------|
| Implementation        | Highly optimized, parallel, robust | Educational, clear, but not optimized |
| Accuracy              | 0.87                     | 0.85               |
| Precision/Recall/F1   | Slightly higher overall  | Slightly lower     |
| Speed                 | Fast (C/C++ backend, parallel) | Slow (Python, no parallelism) |
| Flexibility           | Many hyperparameters, feature importances, etc. | Easy to modify for learning |
| Use Case              | Production, research     | Learning, demonstration |

**Summary:**
- The library-based implementation is more accurate and much faster, benefiting from years of optimization and parallelism.
- The custom implementation is valuable for understanding the algorithm's mechanics and is close in accuracy, but is much slower and less robust.

---

## 4. Conclusion
Both approaches successfully classify brain MRI images using Random Forests. For practical applications, the library-based approach is recommended. The custom implementation is useful for educational purposes and understanding the inner workings of ensemble methods. 