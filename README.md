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

For more details about implementation, please visit this [link](src/models/decision_tree.py)

## Artificial Neural Network (ANN)

The implemented Artificial Neural Network (ANN) is a deep convolutional neural network (CNN) designed for image classification. The architecture consists of four convolutional blocks (feature extractors), followed by a fully connected classifier. This deep CNN is designed to automatically extract hierarchical features from input images, moving from basic edges and textures in the early layers to complex high-level representations in deeper layers. The feature extractor reduces spatial dimensions while increasing depth, capturing meaningful structures in the image. The classifier then transforms the extracted features into a prediction.

Each convolutional block typically consists of five key layers:

1. Convolutional Layer
2. Normalization Layer
3. Activation Layer
4. Pooling Layer
5. Dropout Layer

### Convolution layer

The convolutional layer is responsible for feature extraction. It applies a series of learnable filters (kernels) to the input data, performing convolution operations that capture spatial and hierarchical patterns. By sliding these filters across the input, the layer computes dot products between the kernel values and the corresponding input regions.

Each filter specializes in detecting specific patterns, such as edges, textures, and complex structures, which are essential for accurate image recognition. The output of this operation is referred to as a feature map, which highlights the extracted features for subsequent processing.

### Normalization layer

The normalization layer plays a crucial role in stabilizing the training process and accelerating convergence by ensuring that neuron outputs maintain a standardized distribution. Batch Normalization, which normalizes the outputs within a mini-batch by adjusting their mean and variance, is used as normalization technique in the model. Batch normalization enables the use of higher learning rates, reducing sensitivity to parameter initialization and mitigating the problem of vanishing or exploding gradients

### Activation layer

The activation layer introduces non-linearity into the network, enabling it to learn complex patterns and relationships within the data. ReLu function is used in our model to express that idea.

### Pooling layer

The pooling layer reduces the spatial dimensions of the feature maps while preserving the most significant information. This downsampling process enhances computational efficiency, mitigates overfitting, and ensures robustness to minor spatial variations in the input.

The architecture employs Max Pooling, which selects the maximum value within a defined window (e.g., 2×2). This method retains the most prominent features while discarding less significant information, contributing to effective feature selection.

### Drop layer

The dropout layer is a regularization technique designed to enhance generalization and reduce overfitting in neural networks. Overfitting occurs when the model memorizes training data instead of learning underlying patterns, leading to poor performance on unseen data.

During training, dropout randomly deactivates (i.e., sets to zero) a fraction of neurons within a layer, forcing the network to develop redundant feature representations. This prevents the model from becoming overly dependent on specific pathways and encourages the learning of more robust and distributed feature representations.

For more details about implementation, please visit this [link](src/models/ann.py)

## Genetic Algorithm (GA)

For more details about implementation, please visit this [link](genetic_algorithm/Main.ipynb)

## Naive Bayes

The Naïve Bayes model implemented in this project follows the Gaussian Naïve Bayes (GNB) classification technique, a probabilistic model based on Bayes' theorem with the assumption of feature independence. Each pixel in the input image is treated as an independent feature, and the model estimates the probability that an image belongs to a specific tumor type. Since pixel values are continuous and range from 0 to 255, the Gaussian distribution is chosen to model the likelihood of the features, making it suitable for real-valued inputs. Unlike Bernoulli or Multinomial Naïve Bayes, which are better suited for binary or categorical data, the Gaussian Naïve Bayes classifier assumes that feature values follow a normal distribution for each class. During training, the model learns the mean and variance of pixel values for each class, which are then used to compute class probabilities for unseen data. During inference, the model applies Bayes' theorem to determine the most likely class for a given input based on the learned distributions. This approach is computationally efficient, interpretable, and effective for classification tasks where feature independence is a reasonable assumption.

For more details about implementation, please visit this [link](src/models/naive_bayes.py)

## Bayes Network and Naive Bayes

The Bayesian Network and Naive Bayes implementations leverage probabilistic graphical models to classify brain tumor images using extracted features rather than raw pixel data.

### Feature Extraction

Images are processed through segmentation to extract meaningful features including:

- Geometric properties: area, perimeter, eccentricity, solidity
- Intensity features: mean intensity, contrast, homogeneity
  This feature extraction approach reduces dimensionality while preserving discriminative information

### Model Structure

- Bayesian Network: Implements a directed acyclic graph where features are connected to the tumor type classification
- Naive Bayes: Available as a special configuration where all features are conditionally independent given the tumor type class
- Implemented using the pgmpy library for probabilistic graphical models
- Features automatic discretization of continuous values (configurable number of bins)
- Supports different parameter estimation methods: Maximum Likelihood Estimation (MLE) and Bayesian Estimation with Bayesian Dirichlet equivalent uniform prior

### Visualization

- Network structure visualization shows relationships between variables
- Feature correlation analysis helps understand dependencies between extracted features

### Performance

- Processing in chunks allows handling of large datasets with limited memory
- Includes fallback inference mechanisms when standard inference fails
- Custom probability calculation for both model types ensures robust prediction

For more details about implementation, please visit this [link](src/models/bayes_net.py)

## Support Vector Machine (SVM)

## Dimension Reduced LDA or PCA

## Bagging and Boosting
