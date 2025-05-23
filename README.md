# ML Pupils - Machine Learning Project

Machine learning course @ HCMUT

## Project Team Member and Contribution

- Thai Quang Phat: Data preprocessing, Project Setup
- Nguyen Ngoc Khoi: Implementation of: Naive Bayes, Bayesian Network, Hidden Markov Model, and Hard Margin Support Vector Machine (also support other kernels)
- Nguyen Ngoc Song Thuong: Data Preprocessing, Decision Tree and ANN Implementation
- Pham Duy Tuong Phuoc: Data Preprocessing, Result Analysis, Project Documentation
- Ha Nguyen Bao Phuc: GA Implementation

## Project Description

This project is dedicated to exploring medical classification using machine learning techniques. Our primary objective is to apply multiple machine learning models to a predefined dataset in order to classify medical data accurately and efficiently. To be specific, the work would be about classifying images into different types of tumor.

Through this process, our team seeks to analyze and compare each model based on multiple key metrics, including accuracy, execution time, variation, and overall performance. By evaluating these factors, we hope to uncover insights into the strengths and limitations of different machine learning approaches in handling data.

Ultimately, this project serves as a learning experience for our team, providing us with hands-on exposure to the practical application of machine learning in the computer vision. By systematically experimenting with different models, we aim to deepen our knowledge of their underlying mechanisms and improve our ability to select appropriate techniques for real-world problems.

Machine model our team focuses on include:

- [Decision Tree](#decision-tree)
- [Aritificial Neural Network (ANN)](#artificial-neural-network-ann)
- [Naive Bayesian](#naive-bayesian)
- [Genetic Algorithm (GA)](#genetic-algorithm-ga)
- [Bayesian Network](#bayesian-network)
- [Support Vector Machine (SVM)](#support-vector-machine-svm)
- [Dimension Reduced LDA or PCA](#dimension-reduced-lda-or-pca)
- [Bagging and Boosting](#bagging-and-boosting)

For this study, we will be utilizing the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle. This dataset serves as the foundation for our exploration, enabling us to assess and evaluate the effectiveness of different machine learning approaches in medical image classification.

## How to Use the Project

- For more details about how to install and run the project, please visit [docs/usage.md](docs/usage.md)
- To get a clear understanding of how file system is structured, please visit [docs/implementation.md](docs/implementation.md)

## Decison Tree

## Artificial Neural Network (ANN)

## Genetic Algorithm (GA)

## Bayesian Network and Naive Bayes

### Feature Extraction Process
The Bayesian Network and Naive Bayes models rely on a sophisticated feature extraction pipeline rather than using raw pixel data, making classification more efficient and accurate. Here's how the process works:

### Why Feature Extraction is Necessary
- Dimensionality Reduction: MRI images contain millions of pixels, but most are redundant for classification. Feature extraction reduces this to just 8 meaningful features.
- Focus on Relevant Information: Only certain characteristics of tumors (shape, texture, intensity) are diagnostically relevant.
Robustness: Extracted features are more invariant to variations in image acquisition conditions than raw pixels.
- Interpretability: Features like area and contrast have medical significance, making results more interpretable.

### Detailed Feature Extraction Pipeline
The segment_and_extract_features function implements this multi-stage process:

- Image Preprocessing:
    - Convert to floating-point format for precision
    - Apply histogram equalization to enhance contrast
    - Apply Gaussian smoothing (σ=0.8) to reduce noise
- Tumor Segmentation - Using multiple methods sequentially until successful:
    - Enhanced Otsu thresholding with morphological operations (opening/closing)
    - Adaptive thresholding based on mean intensity
    - Watershed segmentation with distance transform
    - Each step includes border artifact removal and connected component labeling

- Feature Extraction:
    - Geometric Features:
        - Area: Size of tumor region in pixels
        - Perimeter: Boundary length of the tumor
        - Eccentricity: Measure of tumor elongation (0=circle, 1=line)
        - Solidity: Ratio of tumor area to its convex hull area (measure of irregularity)
    - Texture Features using GLCM (Gray Level Co-occurrence Matrix):
        - Contrast: Measures intensity variation between neighboring pixels
        - Homogeneity: Measures texture uniformity
        - Energy: Measures textural uniformity (higher = more uniform)
        - Correlation: Measures linear dependencies between neighboring pixels

### GLCM Analysis In-Depth
The Gray Level Co-occurrence Matrix (GLCM) is a sophisticated texture analysis technique that captures spatial relationships between pixels.

- Matrix Construction: 
    - For each segmented tumor region, the code creates a matrix showing how often specific pairs of pixel intensities occur at particular spatial relationships
    - Spatial Parameters:
            - Distance: The code uses distances of 1 and 3 pixels, analyzing both immediate neighbors and slightly more distant relationships
            - Angles: Four directions (0°, 45°, 90°, 135°) are examined to capture patterns in different orientations
    - Matrix Elements: Each element GLCM[i,j] represents the probability of finding a pixel with intensity i adjacent to a pixel with intensity j. The matrix is normalized (normed=True) so values represent probabilities
    - Result: graycomatrix(roi_valid, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
- Statistical Properties: From this matrix, we derive texture metrics that characterize different tumor types
    - From the GLCM, four key statistical properties are calculated:

    1. **Contrast**: $$\sum_{i,j} (i-j)^2 \times P(i,j)$$
        - Measures local intensity variation
        - High values indicate high contrast between neighboring pixels
        - Relevant for identifying heterogeneous tumor regions

    2. **Homogeneity**: $$\sum_{i,j} \frac{P(i,j)}{1 + |i-j|}$$
        - Measures closeness of element distribution to GLCM diagonal
        - Higher values indicate more uniform textures
        - Useful for differentiating smooth vs. irregular tumor surfaces

    3. **Energy**: $$\sum_{i,j} P(i,j)^2$$
        - Sum of squared elements in the GLCM
        - Measures textural uniformity (higher = more uniform)
        - Helps identify repeating texture patterns

    4. **Correlation**: $$\sum_{i,j} \frac{(i-\mu_i)(j-\mu_j)P(i,j)}{\sigma_i\sigma_j}$$
        - Measures linear dependencies between neighboring pixels
        - Values range from -1 to 1
        - Indicates how predictable pixel relationships are
- Different tumor types exhibit characteristic texture patterns:
    - Gliomas often show heterogeneous textures (lower homogeneity, higher contrast)
    - Meningiomas typically have more uniform textures (higher energy)
    - Metastases may have distinctive correlation patterns

### Fallback Mechanism
If segmentation fails (no tumor found or segmentation issues):
- The code calculates global image statistics as approximations
- Contrast is derived from normalized standard deviation
- Homogeneity is calculated from histogram entropy
- Area and perimeter are approximated from image dimensions

### Model Implementation
- Bayesian Network: Models probabilistic relationships between extracted features and tumor classes
- Naive Bayes: Special case where all features are conditionally independent given the class
Both models use discretized versions of the extracted continuous features. Parameter estimation uses either Maximum Likelihood or Bayesian estimation techniques.

### Evaluation
#### Result
```
              precision    recall  f1-score   support

      glioma       0.74      0.54      0.62       405
  meningioma       0.44      0.73      0.55       300
     notumor       1.00      1.00      1.00       300
   pituitary       0.32      0.23      0.26       306

    accuracy                           0.61      1311
   macro avg       0.63      0.62      0.61      1311
weighted avg       0.63      0.61      0.61      1311
```

#### Overall Performance
- Accuracy: 61% - The model achieves moderate performance overall, substantially better than random guessing (25% for 4 classes)
- Consistency: The weighted precision, recall, and F1-score all hover around 61-63%, indicating balanced performance across metrics

#### Class-Specific Analysis
- "No Tumor" Classification: Perfect performance (100% precision, recall) - The model excels at distinguishing between tumor and non-tumor cases
- Glioma Detection: Good precision (74%) but moderate recall (54%) - When the model predicts glioma, it's usually correct, but it misses nearly half of actual glioma cases
- Meningioma Detection: Lower precision (44%) with good recall (73%) - The model identifies most meningioma cases but frequently misclassifies other tumors as meningioma
- Pituitary Detection: Poor performance overall (F1=26%) - The model struggles significantly with this tumor type

#### Limitations and Areas for Improvement
- Pituitary classification failure - The model particularly struggles with pituitary tumors, suggesting:
    - The extracted GLCM features may not capture distinctive characteristics of pituitary tumors
    - There might be class imbalance issues affecting the learning process
- Feature engineering opportunities:
    - Additional texture or shape features might help differentiate between tumor types
    - More sophisticated segmentation techniques could improve feature quality
    - Consider domain-specific features based on medical knowledge

## Hidden Markov Model

The Hidden Markov Model (HMM) implementation uses a Gaussian HMM approach for tumor classification. This model is particularly effective for capturing temporal and sequential patterns in the image features.

### Implementation Details
- **Model Architecture**: Uses Gaussian HMM with configurable number of states (default: 4 states)
- **Feature Extraction**: Implements sophisticated feature extraction pipeline including:
  - Image preprocessing with histogram equalization:
    ```python
    # Histogram equalization using skimage
    img_eq = equalize_hist(img)  # Normalizes image to [0,1] range
    ```
  - Gaussian smoothing for noise reduction:
    ```python
    # Gaussian smoothing with σ=0.8
    img_smooth = gaussian(img_uint8, sigma=0.8)
    ```
  - Feature extraction focusing on geometric and texture characteristics:
    - Geometric features: area, perimeter, eccentricity, solidity
    - GLCM (Gray-Level Co-occurrence Matrix) texture features:
      ```python
      # GLCM calculation with multiple distances and angles
      glcm = graycomatrix(roi, 
                         distances=[1, 3],
                         angles=[0, π/4, π/2, 3π/4],
                         levels=256,
                         symmetric=True,
                         normed=True)
      ```
      - Contrast: $$\sum_{i,j} (i-j)^2 P(i,j)$$
      - Homogeneity: $$\sum_{i,j} \frac{P(i,j)}{1 + |i-j|}$$
      - Energy: $$\sum_{i,j} P(i,j)^2$$
      - Correlation: $$\sum_{i,j} \frac{(i-\mu_i)(j-\mu_j)P(i,j)}{\sigma_i\sigma_j}$$

- **Training Process**:
  - Trains separate HMM models for each tumor class
  - Uses maximum likelihood estimation for parameter learning:
    - Initial state probabilities: $$\pi_i = P(q_1 = i)$$
    - Transition probabilities: $$a_{ij} = P(q_{t+1} = j | q_t = i)$$
    - Emission probabilities: $$b_j(k) = P(o_t = k | q_t = j)$$
  - Implements automatic state number adjustment based on class sample size

### Technical Specifications
- **Feature Processing**:
  - Automatic handling of NaN and infinite values using numpy's nan_to_num
  - Feature normalization and standardization using scikit-image
  - Support for batch processing of images using tqdm progress bars
- **Model Training**:
  - Adaptive state number adjustment based on class sample size
  - Robust error handling for edge cases
  - Automatic model validation during training
- **Evaluation Process**:
  - Log-likelihood based classification:
    $$P(O|λ) = \sum_{Q} P(O,Q|λ) = \sum_{Q} P(O|Q,λ)P(Q|λ)$$
  - Batch processing for efficient evaluation
  - Comprehensive error handling and logging

## Support Vector Machine (SVM)

The SVM implementation provides a robust approach to tumor classification using both linear and non-linear kernels.

### Implementation Details
- **Model Architecture**: Uses scikit-learn's SVC with configurable parameters
- **Feature Processing**:
  - Automatic image flattening for feature vector creation:
    ```python
    # Flattening (N, H, W, C) → (N, H*W*C)
    n_samples = X.shape[0]
    X = X.reshape(n_samples, -1)
    ```
  - Support for both raw pixel values and extracted features
  - Handles multi-dimensional input data using numpy's reshape operations
- **Training Process**:
  - Supports multiple kernel types:
    - Linear: $$K(x,y) = x^T y$$
    - RBF: $$K(x,y) = \exp(-\gamma ||x-y||^2)$$
    - Polynomial: $$K(x,y) = (\gamma x^T y + r)^d$$
  - Implements probability estimates using Platt scaling
  - Uses efficient optimization algorithms from scikit-learn's SVC implementation
- **Key Parameters**:
  - `C`: Regularization parameter (default: 1.0)
  - `kernel`: Kernel type (default: 'rbf')
  - `gamma`: Kernel coefficient (default: 'scale')
  - `probability`: Enabled for probability estimates

### Technical Specifications
- **Data Preprocessing**:
  - Automatic image flattening: (N, H, W, C) → (N, H*W*C) using numpy.reshape
  - Batch size: Configurable (default: 128 samples) using tqdm for progress tracking
  - Memory-efficient processing using scikit-learn's SVC implementation
- **Model Training**:
  - Probability estimation enabled by default using Platt scaling
  - Support for both binary and multi-class classification using One-vs-Rest
  - Efficient batch processing implementation using scikit-learn's batch_size parameter
- **Evaluation Process**:
  - Batch-wise prediction for memory efficiency:
    ```python
    for i in range(0, n_samples, batch_size):
        batch_X = X[i:min(i + batch_size, n_samples)]
        batch_y_pred = model.predict(batch_X)
    ```
  - Probability estimates for each class using predict_proba
  - Comprehensive error handling and logging

## Kernel SVM

## Dimension Reduction: LDA / PCA

## Bagging and Boosting
