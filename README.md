# ML Pupils - Machine Learning Project

Machine learning course @ HCMUT

## Project Team Member and Contribution

- Thai Quang Phat: Data preprocessing, Project Setup
- Nguyen Ngoc Khoi: Naive Bayes and Bayesian Network Implementation
- Nguyen Ngoc Song Thuong: Data Preprocessing, Decision Tree and ANN Implementation
- Pham Duy Tuong Phuoc: Data Preprocessing, Result Analysis, Project Documentation
- Ha Nguyen Bao Phuc: GA Implementation

## Project Description

This project is dedicated to exploring medical classification using machine learning techniques. Our primary objective is to apply multiple machine learning models to a predefined dataset in order to classify medical data accurately and efficiently. To be specific, the work would be about classifying images into different types of tumor.

Through this process, our team seeks to analyze and compare each model based on multiple key metrics, including accuracy, execution time, variation, and overall performance. By evaluating these factors, we hope to uncover insights into the strengths and limitations of different machine learning approaches in handling data.

Ultimately, this project serves as a learning experience for our team, providing us with hands-on exposure to the practical application of machine learning in the computer vision. By systematically experimenting with different models, we aim to deepen our knowledge of their underlying mechanisms and improve our ability to select appropriate techniques for real-world problems.

Machine model our team focuses on include:

- [Decision Tree](#decison-tree)
- [Aritificial Neural Network (ANN)](#artificial-neural-network-ann)
- [Naive Bayes](#naive-bayes)
- [Genetic Algorithm (GA)](#genetic-algorithm-ga)
- [Bayes Network](#bayes-network-and-naive-bayes)
- [Support Vector Machine (SVM)](#support-vector-machine-svm)
- [Dimension Reduced LDA or PCA](#dimension-reduced-lda-or-pca)
- [Bagging and Boosting](#bagging-and-boosting)

For this study, we will be utilizing the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle. This dataset serves as the foundation for our exploration, enabling us to assess and evaluate the effectiveness of different machine learning approaches in medical image classification.

## How to Use the Project

- For more details about how to install and run the project, please visit [docs/usage.md](docs/usage.md)
- To get a clear understanding of how file system is structured, please visit [docs/implementation.md](docs/implementation.md)

## Decison Tree

The Decision Tree model implemented follows a supervised learning approach for classification tasks. The model is built using Scikit-learn’s DecisionTreeClassifier, which constructs a tree-based structure for decision-making. The input dataset consists of feature vectors extracted from images, which are flattened into a one-dimensional representation before training. The decision tree employs the Gini impurity criterion to measure the quality of splits, ensuring that each node partitions the data to maximize class purity. The tree is trained recursively by selecting the optimal feature at each node, splitting the data until a stopping criterion is met, such as reaching pure leaf nodes or a predefined depth. During inference, the model traverses the tree based on the feature values of an input sample, following the learned decision boundaries to assign a class label.

We use `DecisionTreeClassifer` from sklearn.tree with grid search on 5-fold cross validation.
The hyperparameters that need tuning include:

- `criterion`: ['gini', 'entropy']. Gini index one of the feature selection metrics that is used for building binary tree with faster computation, while Entropy is suitable for multi-class classification problem. We not tuning with 'log_loss' as it is more suitable for binary classification, which is not the case.
- `max_features`: [50..10000]. As there are 256x256 features in an input array, the time complexity when choosing the best split is huge. To increase training time while maintain the model performance, we try to tune max_features to different values that is smaller than the original space.
- `min_samples_leaf`: [50,100]. For generalization, we want to optimize the minimum samples at the leaf node, which is equivalent to performing a post pruning.
- Other pruning-related hyperparameters such as `max_depth` is not used as we run on multiple trials and the max_depth in our case is not very deep.

For grid search, we using 'recall_macro' scoring strategy. The reason is that our task is to classify medical image, focusing on the increasing the number of correct prediction for having brain tumor, for which `recall` is most valuable metrics. Between 'recall_macro' and 'recall_micro', we use 'recall_macro' as our datasets split into 4 equal categories of brain tumor.

The result of each running time is logged in results/log.
**Analysis of Decision Tree Model Performance**

### **Model Details**

- **Tree Depth:** 17
- **Total Nodes:** 369
- **Leaf Nodes:** 185
- **Number of Features Used:** 65,536
- **Hyperparameters Optimized:**
  - **Criterion:** Gini
  - **Max Features:** 10,000
  - **Min Samples per Leaf:** 20

### **Performance Metrics**

- **Overall Accuracy:** 53%
- **Macro Average F1-score:** 0.51
- **Weighted Average F1-score:** 0.52

#### **Class-wise Performance**

| Class | Precision | Recall | F1-Score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.47      | 0.44   | 0.46     | 300     |
| 1     | 0.36      | 0.33   | 0.35     | 306     |
| 2     | 0.67      | 0.72   | 0.69     | 405     |
| 3     | 0.53      | 0.55   | 0.54     | 300     |

### **Observations & Issues**

- The model achieves an overall accuracy of 53%, indicating limited predictive power.
- Class 2 performs significantly better than others, suggesting potential data distribution issues.
- The model uses 65,536 features, which can lead to overfitting and reduced interpretability.
- Despite feature selection using max_features = 10,000, the performance remains low, suggesting that many features might be irrelevant or redundant.
- A tree depth of 17 with 369 nodes suggests a complex model that may be capturing noise rather than generalizable patterns.
- The best cross-validation score was 48.7%, indicating that even with optimal parameters, the model struggles to generalize effectively.

### **Use Case Fit Conclusion**

The current Decision Tree model demonstrates limited effectiveness in brain tumor MRI classification due to high feature dimensionality, class imbalance, and potential overfitting. While it may provide some baseline insights, its relatively low accuracy and recall suggest that it is not well-suited for high-stakes medical applications.

For more details about implementation, please visit this [link](src/models/decision_tree/)

## Artificial Neural Network (ANN)

### Customed ANN

The implemented Artificial Neural Network (ANN) is a convolutional neural network (CNN) designed for image classification, consisting of multiple feature extraction layers followed by a classification module. The model follows a hierarchical approach, progressively refining extracted features to improve classification accuracy.

#### Architecture Overview

The ANN consists of four feature extraction blocks, each containing convolutional layers, batch normalization, activation functions, and pooling operations. These layers extract spatial features from input images, capturing increasingly complex patterns as the depth of the network increases. The extracted features are then processed by a classification module that predicts the final output.

| Layer        | Operation                                                                                                                                        | Output Size (C × H × W) |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------- |
| _Input_      | -                                                                                                                                                | 1 × 256 × 256           |
| _Feature1_   | `Conv2D(1 → 6, kernel=5, stride=1, padding=2)` → `BatchNorm2d(6)` → `ReLU` → `MaxPool2d(2×2, stride=2)`                                          | 6 × 128 × 128           |
| _Feature2_   | `Conv2D(6 → 16)` → `BatchNorm2d(16)` → `ReLU` → `MaxPool2d(2×2)` → `Dropout(0.1)`                                                                | 16 × 64 × 64            |
| _Feature3_   | `Conv2D(16 → 32)` → `BatchNorm2d(32)` → `ReLU` → `MaxPool2d(2×2)` → `Dropout(0.1)`                                                               | 32 × 32 × 32            |
| _Feature4_   | `Conv2D(32 → 64)` → `BatchNorm2d(64)` → `ReLU` → `MaxPool2d(2×2)` → `Dropout(0.2)`                                                               | 64 × 16 × 16            |
| _Classifier_ | `Conv2D(64 → 120)` → `BatchNorm2d(120)` → `ReLU` → `AdaptiveAvgPool(1×1)` → `Flatten` → `Linear(120 → 84)` → `ReLU` → `Linear(84 → num_classes)` | num_classes (e.g., 4)   |

#### Key Components of the Model

##### Convolution layer

The convolutional layer is responsible for feature extraction. It applies a series of learnable filters (kernels) to the input data, performing convolution operations that capture spatial and hierarchical patterns. By sliding these filters across the input, the layer computes dot products between the kernel values and the corresponding input regions.

Each filter specializes in detecting specific patterns, such as edges, textures, and complex structures, which are essential for accurate image recognition. The output of this operation is referred to as a feature map, which highlights the extracted features for subsequent processing.

##### Normalization layer

The normalization layer plays a crucial role in stabilizing the training process and accelerating convergence by ensuring that neuron outputs maintain a standardized distribution. Batch Normalization, which normalizes the outputs within a mini-batch by adjusting their mean and variance, is used as normalization technique in the model. Batch normalization enables the use of higher learning rates, reducing sensitivity to parameter initialization and mitigating the problem of vanishing or exploding gradients

##### Activation layer

The activation layer introduces non-linearity into the network, enabling it to learn complex patterns and relationships within the data. ReLu function is used in our model to express that idea.

##### Pooling layer

The pooling layer reduces the spatial dimensions of the feature maps while preserving the most significant information. This downsampling process enhances computational efficiency, mitigates overfitting, and ensures robustness to minor spatial variations in the input.

The architecture employs Max Pooling, which selects the maximum value within a defined window (e.g., 2×2). This method retains the most prominent features while discarding less significant information, contributing to effective feature selection.

##### Drop layer

The dropout layer is a regularization technique designed to enhance generalization and reduce overfitting in neural networks. Overfitting occurs when the model memorizes training data instead of learning underlying patterns, leading to poor performance on unseen data.

During training, dropout randomly deactivates (i.e., sets to zero) a fraction of neurons within a layer, forcing the network to develop redundant feature representations. This prevents the model from becoming overly dependent on specific pathways and encourages the learning of more robust and distributed feature representations.

#### **Performance Metrics**

- **Overall Accuracy:** 89%
- **Macro Average F1-score:** 0.89
- **Weighted Average F1-score:** 0.89

##### **Class-wise Performance**

| Class | Precision | Recall | F1-Score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.81      | 0.72   | 0.76     | 306     |
| 1     | 0.84      | 1.00   | 0.92     | 405     |
| 2     | 0.96      | 0.98   | 0.97     | 300     |
| 3     | 0.98      | 0.82   | 0.90     | 300     |

#### **Observations & Issues**

- The ANN model achieves an accuracy of **89%**, demonstrating strong classification capability.
- High precision and recall for most classes indicate reliable performance, though Class 0 shows a slightly lower recall (0.72), suggesting some misclassifications.
- The model provides a balanced classification across all categories with consistent F1-scores.
- Performance could be further improved by fine-tuning hyperparameters, increasing dataset size, or implementing data augmentation techniques.

#### **Use Case Fit Conclusion**

The ANN model is well-suited for brain tumor MRI classification, achieving high accuracy and reliable class-wise performance. Its ability to learn spatial and textural patterns from MRI images makes it a strong candidate for deployment in medical imaging applications. The model demonstrates balanced precision and recall across multiple tumor classes, ensuring robust classification. Further improvements can be explored through hyperparameter tuning, increased dataset diversity, and advanced techniques such as transfer learning or ensemble methods to further enhance performance and generalization.

### ResNet18 architecture

This project implements a deep convolutional neural network (CNN) based on the ResNet18 architecture, replicating the model presented in “Deep Residual Learning for Image Recognition”. ResNet18 is a state-of-the-art deep learning model designed for image classification, incorporating residual connections to address the vanishing gradient problem and improve training efficiency in deep networks. The proposed model is specifically tailored for grayscale brain MRI image classification, utilizing four residual convolutional blocks for feature extraction, followed by a fully connected classifier for prediction. The residual connections allow efficient gradient propagation, thereby enhancing the model’s training stability and performance.

#### Performance Comparison with Standard CNN Models

The effectiveness of the ResNet18-based model was evaluated in comparison to a conventional CNN that employs only convolutional and max pooling layers. The results indicate that:

- The conventional CNN achieved an accuracy of 83% in brain MRI classification.
- The ResNet18-based model demonstrated a 94% accuracy, highlighting the advantages of residual connections in deep neural networks.

#### **Layer-wise Output Size Calculation**

| **Layer**                                | **Operation**                       | **Output Size (C × H × W)**  |
| ---------------------------------------- | ----------------------------------- | ---------------------------- |
| **Input**                                | -                                   | **1 × 256 × 256**            |
| **Conv1 (7×7, stride=2, padding=3)**     | Conv2D(1 → 64)                      | **64 × 128 × 128**           |
| **MaxPool (3×3, stride=2, padding=1)**   | Downsampling                        | **64 × 64 × 64**             |
| **Layer1 (2 Residual Blocks, stride=1)** | Conv2D(64 → 64)                     | **64 × 64 × 64**             |
| **Layer2 (2 Residual Blocks, stride=2)** | Conv2D(64 → 128)                    | **128 × 32 × 32**            |
| **Layer3 (2 Residual Blocks, stride=2)** | Conv2D(128 → 256)                   | **256 × 16 × 16**            |
| **Layer4 (2 Residual Blocks, stride=2)** | Conv2D(256 → 512)                   | **512 × 8 × 8**              |
| **AvgPool (1×1 Adaptive)**               | Global pooling                      | **512 × 1 × 1**              |
| **Flatten**                              | -                                   | **512**                      |
| **FC (Fully Connected Layer)**           | Output **(e.g., num_classes=1000)** | **1000 (or custom classes)** |

---

#### **Performance Metrics**

- **Overall Accuracy:** 94%
- **Macro Average F1-score:** 0.94
- **Weighted Average F1-score:** 0.94

##### **Class-wise Performance**

| Class | Precision | Recall | F1-Score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.97      | 0.98   | 0.98     | 405     |
| 1     | 0.97      | 0.91   | 0.94     | 300     |
| 2     | 0.95      | 0.94   | 0.95     | 300     |
| 3     | 0.88      | 0.92   | 0.90     | 306     |

#### **Observations & Issues**

- The ResNet18 model significantly outperforms both the Decision Tree and Gaussian Naïve Bayes models, achieving an accuracy of 94%.
- High precision and recall across all classes indicate strong generalization and robust feature extraction.
- Class 3 shows slightly lower precision (0.88) compared to other classes, suggesting potential misclassifications.
- The model effectively learns complex spatial features from MRI images, overcoming the limitations seen in previous models.

#### **Use Case Fit Conclusion**

The ResNet18 model is well-suited for brain tumor MRI classification, demonstrating superior accuracy and class balance compared to other models. Its ability to capture spatial features makes it a strong candidate for deployment in medical imaging applications. Further improvements can be explored through fine-tuning, data augmentation, and deeper architectures like Convolutional Neural Networks (CNNs) for even higher performance.

For more details about implementation, please visit this [link](src/models/ann/)
The training process includes early stopping and saving best-performance model based on validation accuracy.

## Genetic Algorithm (GA)
> **Note:** The following explanation is based on the notebook `notebooks/assignment1/genetic_algorithm.ipynb`. However, the actual implementation in `src/models/genetic_algorithm/` follows a structured pipeline for processing, training, and evaluation.
   
In this usecase, Genetic Algorithm (GA) is used to optimize the performance of an Artificial Neural Networks (ANNs).
The implementation is based on PyTorch in order to utilizes its GPU computing.

### Data preparation and preprocessing

The original dataset is downloaded and stored in the directory `dataset/Original`. After preprocessing, the new dataset is stored at `dataset/AfterPreprocess`.

**Dataset structure**

```
dataset
|-- Original
|        |-- Training
|        |       |-- glioma     (Tr_gl_xxxx.jpg)    --> 1321 images
|        |       |-- meningioma (Tr_me_xxxx.jpg)    --> 1339 images
|        |       |-- notumor    (Tr_no_xxxx.jpg)    --> 1595 images
|        |       |-- pituitary  (Tr_pi_xxxx.jpg)    --> 1457 images
|        |-- Testing
|        |       |-- glioma     (Te_gl_xxxx.jpg)    --> 300 images
|        |       |-- meningioma (Te_me_xxxx.jpg)    --> 306 images
|        |       |-- notumor    (Te_no_xxxx.jpg)    --> 405 images
|        |       |-- pituitary  (Te_pi_xxxx.jpg)    --> 300 images
|-- AfterPreprocess
|        |-- Training
|        |       |-- glioma     (Tr_gl_xxxx.jpg)    --> 1321 images
|        |       |-- meningioma (Tr_me_xxxx.jpg)    --> 1339 images
|        |       |-- notumor    (Tr_no_xxxx.jpg)    --> 1595 images
|        |       |-- pituitary  (Tr_pi_xxxx.jpg)    --> 1457 images
|        |-- Testing
|        |       |-- glioma     (Te_gl_xxxx.jpg)    --> 300 images
|        |       |-- meningioma (Te_me_xxxx.jpg)    --> 306 images
|        |       |-- notumor    (Te_no_xxxx.jpg)    --> 405 images
|        |       |-- pituitary  (Te_pi_xxxx.jpg)    --> 300 images
|        |-- augmented_img_paths.json
```

The preprocessing phase is handled by class **ImgPreprocess**, including:

- Crop image to center the brain.
- Augment the image with the ratio 0.25 (meaning that 25% image of each class will be augmented) and store the path of the augmented images in `dataset/AfterPreprocess/augmented_img_paths.json`
- Crop the image to the size (255, 255)

### Dataset and Dataloader

The dataset is handled by class **BrainTumorDataset** which utilizes **torch.utils.data.Dataset**. The dataloader is handled by class **Loader** which utilizes **torch.utils.data.Dataset**. The batch size is set to 32 and 20% of the training set is kept for validation.

### Dynamic Neural Networks

The general architectures of the neural networks used in this use cases is that there are several Convolutional Neural Networks (CNNs) are stacked together, followed by some Fully Connected Layers to produce the output.

This architecture is handled by the class **DynamicNN**. With this implementation, it allows the architecture of the networks to grow or shrink freely as well as the learning rate or its activation fucntion, depending on the optimization of the genetic algorithm.

### Genetic Algorithm

The genetic algorithm plays as an optimizer to optimize the performance of the DynamicNN. This optimization is based on randomly growing or shrinking the architecture of the DynamicNN, changing the activation function or finding another suitable value for learning rate. All of these things are handled by class **GAOptimizer**.

### **Performance Metrics**

- **Fitness**: 0.9483
- **Final Test Accuracy**: 0.9860

### **Observations & Issues**

- With a fitness score of 0.9483 and a final test accuracy of 98.60%, the ANN, guided by GA, can effectively learn meaningful feature representations and achieve superior classification performance.
- The training loss values, which progressively decrease across epochs, suggest stable and efficient learning

### **Use case fit conclusion**

The Genetic Algorithm (GA)-optimized Artificial Neural Network (ANN) demonstrates strong effectiveness in brain tumor MRI classification by dynamically evolving an optimal network topology. By systematically optimizing key parameters such as layer depth, neuron distribution, and activation functions, the GA enables the ANN to efficiently learn complex spatial patterns inherent in MRI images. The model’s high test accuracy and stable learning process indicate its robustness, making it well-suited for high-stakes medical applications where precision is critical.

For more details about implementation, please visit this [link](notebooks/assignment1/genetic_algorithm.ipynb)

## Naive Bayes

The Naïve Bayes model implemented in this project follows the Gaussian Naïve Bayes (GNB) classification technique, a probabilistic model based on Bayes' theorem with the assumption of feature independence. Each pixel in the input image is treated as an independent feature, and the model estimates the probability that an image belongs to a specific tumor type. Since pixel values are continuous and range from 0 to 255, the Gaussian distribution is chosen to model the likelihood of the features, making it suitable for real-valued inputs. Unlike Bernoulli or Multinomial Naïve Bayes, which are better suited for binary or categorical data, the Gaussian Naïve Bayes classifier assumes that feature values follow a normal distribution for each class. During training, the model learns the mean and variance of pixel values for each class, which are then used to compute class probabilities for unseen data. During inference, the model applies Bayes' theorem to determine the most likely class for a given input based on the learned distributions. This approach is computationally efficient, interpretable, and effective for classification tasks where feature independence is a reasonable assumption.

The Gaussian Naïve Bayes model is not well-suited for classifying brain tumors from MRI images, as evidenced by its highest accuracy score 60%. This could be due the fact that this model assumes independence among features, which is rarely the case in complex medical imaging data where pixel intensities and spatial relationships play a crucial role in classification. Naïve Bayes is typically effective for text classification and simpler datasets but lacks the ability to capture the intricate, hierarchical patterns present in MRI scans. Additionally, its reliance on strong independence assumptions leads to suboptimal performance when dealing with correlated features, which are common in image data.

For more details about implementation, please visit this [link](src/models/naive_bayes/)

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
    - Matrix Elements: Each element ```GLCM[i,j]``` represents the probability of finding a pixel with intensity i adjacent to a pixel with intensity j. The matrix is normalized (normed=True) so values represent probabilities
    - Result: ```graycomatrix(roi_valid, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)```
- Statistical Properties: From this matrix, we derive texture metrics that characterize different tumor types
    - From the GLCM, four key statistical properties are calculated:

    1. **Contrast**: `Σ(i,j) (i-j)² × P(i,j)`
        - Measures local intensity variation
        - High values indicate high contrast between neighboring pixels
        - Relevant for identifying heterogeneous tumor regions

    2. **Homogeneity**: `Σ(i,j) P(i,j) / (1 + |i-j|)`
        - Measures closeness of element distribution to GLCM diagonal
        - Higher values indicate more uniform textures
        - Useful for differentiating smooth vs. irregular tumor surfaces

    3. **Energy**: `Σ(i,j) P(i,j)²`
        - Sum of squared elements in the GLCM
        - Measures textural uniformity (higher = more uniform)
        - Helps identify repeating texture patterns

    4. **Correlation**: `Σ(i,j) ((i-μi)(j-μj)P(i,j))/(σiσj)`
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

For more details about implementation, please visit this [link](src/models/bayes_net/)

## Support Vector Machine (SVM)

## Dimension Reduced LDA or PCA

## Bagging and Boosting
