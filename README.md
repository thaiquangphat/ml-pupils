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

## Support Vector Machine (SVM)

## Dimension Reduced LDA or PCA

## Bagging and Boosting
