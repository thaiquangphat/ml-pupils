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
Both models use discretized versions of the extracted continuous features. Parameter estimation uses either Maximum Likelihood or Bayesian estimation techniques


## Support Vector Machine (SVM)

## Dimension Reduced LDA or PCA

## Bagging and Boosting
