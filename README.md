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
The general architectures of the neural networks used in this use cases is that there are several Convolutional Neural Networks (CNNs) are stacked  together, followed by some Fully Connected Layers to produce the output. 

This architecture is handled by the class **DynamicNN**. With this implementation, it allows the architecture of the networks to grow or shrink freely as well as the learning rate or its activation fucntion, depending on the optimization of the genetic algorithm.

### Genetic Algorithm
The genetic algorithm plays as an optimizer to optimize the performance of the DynamicNN. This optimization is based on randomly growing or shrinking the architecture of the DynamicNN, changing the activation function or finding another suitable value for learning rate. All of these things are handled by class **GAOptimizer**.



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
