# SVM with Polynomial Kernel (Degree 2) Report

## Overview
This report summarizes the implementation and evaluation of the Support Vector Machine (SVM) algorithm with a polynomial kernel of degree 2 for image classification, as presented in the `svm_kernel_polydeg2.ipynb` notebook. The project includes two approaches:
- Using the `SVC` from scikit-learn (library-based)
- A custom re-implementation of SVM with polynomial kernel (degree 2) from scratch (with One-vs-Rest for multiclass)

Both approaches use features extracted from images (via a deep learning model) and are evaluated on the same dataset.

---

## 1. Polynomial Kernel Formula
The polynomial kernel of degree 2 is defined as:

\[
K(x, y) = (x \cdot y + 1)^2
\]

where \(x\) and \(y\) are feature vectors.

---

## 2. SVM Using scikit-learn

### Implementation Summary
- **Feature Extraction:** Features are extracted from images using a pre-trained VGG19 model and further processed with PCA for dimensionality reduction.
- **Preprocessing:**
  - Features are scaled using `StandardScaler`.
  - PCA is applied to reduce dimensionality (up to 64 components).
- **Model:**
  - `SVC(kernel="poly", degree=2, C=1.0, probability=True)` from scikit-learn is used.
  - The model is trained on the processed features.

```python
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
train_features_pca = pca.fit_transform(train_features_scaled)
svm_model = SVC(kernel="poly", degree=2, C=1.0, probability=True)
svm_model.fit(train_features_pca, train_labels)
```

### Results
- **Accuracy:** 0.8185 (81.85%)
- **Classification Report:**
```
              precision    recall  f1-score   support

      glioma       0.81      0.68      0.74       300
  meningioma       0.63      0.73      0.68       306
     notumor       0.95      0.91      0.93       405
   pituitary       0.88      0.93      0.90       300

    accuracy                           0.82      1311
   macro avg       0.82      0.81      0.81      1311
weighted avg       0.83      0.82      0.82      1311
```

---

## 3. Custom SVM Implementation (Polynomial Kernel Degree 2)

### Implementation Summary
- **Kernel:**
  - Custom `SVMPolyDegree2` class implements the kernel: \(K(x, y) = (x \cdot y + 1)^2\)
  - Uses quadratic programming (via `cvxopt`) to solve the dual problem.
- **Multiclass:**
  - `MultiClassSVM_OvR` implements One-vs-Rest for multiclass classification.

### Mathematical Formulation

#### 1. Initialization
The model is initialized with a regularization parameter C:
\[
\text{Initialize with } C = 1.0 \text{ (default)}
\]

#### 2. Polynomial Kernel Function
The polynomial kernel of degree 2 is implemented as:
\[
K(x, y) = (x \cdot y + 1)^2
\]
where:
- \(x\) and \(y\) are feature vectors
- \(x \cdot y\) is the dot product
- The kernel maps the input space to a higher-dimensional feature space

#### 3. Training Process (Dual Problem)
The SVM training solves the following quadratic programming problem:

\[
\min_{\alpha} \frac{1}{2}\alpha^T P \alpha + q^T \alpha
\]

Subject to:
\[
G \alpha \leq h
\]
\[
A \alpha = b
\]

Where:
- \(P = (y_i y_j K(x_i, x_j))_{i,j}\) is the kernel matrix
- \(q = -1\) (vector of ones)
- \(G = \begin{bmatrix} -I \\ I \end{bmatrix}\) (constraints matrix)
- \(h = \begin{bmatrix} 0 \\ C \end{bmatrix}\) (constraints vector)
- \(A = y^T\) (equality constraint)
- \(b = 0\) (equality constraint)

#### 4. Support Vector Selection
After solving the QP problem, support vectors are selected where:
\[
\alpha_i > 1e^{-5}
\]

#### 5. Bias Calculation
The bias term is calculated as:
\[
b = \frac{1}{n_{sv}} \sum_{k=1}^{n_{sv}} (y_k - \sum_{i=1}^{n_{sv}} \alpha_i y_i K(x_k, x_i))
\]
where:
- \(n_{sv}\) is the number of support vectors
- \(x_k, y_k\) are support vectors and their labels
- \(\alpha_i\) are the Lagrange multipliers
- \(K(x_k, x_i)\) is the kernel function

#### 6. Decision Function
For prediction, the decision function is:
\[
f(x) = \text{sign}(\sum_{i=1}^{n_{sv}} \alpha_i y_i K(x, x_i) + b)
\]

#### 7. Multiclass Extension (One-vs-Rest)
For multiclass classification, the model uses One-vs-Rest approach:
- For each class \(c\):
  - Create binary labels: \(y_i = 1\) if \(x_i \in c\), else \(y_i = -1\)
  - Train a binary SVM classifier
  - Store the decision function \(f_c(x)\)
- Final prediction: \(\arg\max_c f_c(x)\)

### Results
- **Accuracy:** 0.82 (82%)
- **Classification Report:**
```
              precision    recall  f1-score   support

      glioma       0.84      0.71      0.77       300
  meningioma       0.65      0.71      0.68       306
     notumor       0.92      0.92      0.92       405
   pituitary       0.88      0.93      0.90       300

    accuracy                           0.82      1311
   macro avg       0.82      0.82      0.82      1311
weighted avg       0.83      0.82      0.82      1311
```

### Analysis
Both implementations achieved similar overall accuracy of around 82%. The custom implementation showed slightly better performance for glioma classification (84% precision vs 81%) but slightly worse for meningioma (65% precision vs 63%). Both models performed exceptionally well on the "notumor" and "pituitary" classes, with precision and recall above 90%.

The results demonstrate that the polynomial kernel of degree 2 is effective for this classification task, with both implementations showing strong performance across all classes. The custom implementation's performance being comparable to scikit-learn's implementation validates our implementation approach.

---

## 4. Comparison and Discussion

| Aspect                | scikit-learn SVM         | Custom SVM (Poly Deg 2) |
|-----------------------|-------------------------|-------------------------|
| Implementation        | Highly optimized, robust| Educational, clear, but not optimized |
| Accuracy              | (fill in)               | (fill in)               |
| Precision/Recall/F1   | (fill in)               | (fill in)               |
| Speed                 | Fast (C/C++ backend)    | Slow (Python, no parallelism) |
| Flexibility           | Many hyperparameters    | Easy to modify for learning |
| Use Case              | Production, research    | Learning, demonstration  |

**Summary:**
- The library-based implementation is more accurate and much faster, benefiting from years of optimization.
- The custom implementation is valuable for understanding the algorithm's mechanics and is close in accuracy, but is much slower and less robust.

---

## 5. Conclusion
Both approaches successfully classify brain MRI images using SVMs with a polynomial kernel of degree 2. For practical applications, the library-based approach is recommended. The custom implementation is useful for educational purposes and understanding the inner workings of kernel methods. 