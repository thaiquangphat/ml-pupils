# ASSIGNMENT 02: LOGISTIC REGRESSION AND PCA

In this assignment, we reuse the dataset that was cleaned and pre-processed in the assignment 01. 

Recall the structure of the dataset:

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

## I - LOGISTIC REGRESSION
In this section, we built a simple Logistic regression model to classify only 2 of 4 classes of the dataset, `notumor` and `glioma`. However, in order to have better performance, we use 2 CNN layers. Then, the Linear classifier follows and produces the output.

The implementation is based on PyTorch, the device used in the training is `cuda`.

The model is trained for 40 epoches. The plots of training loss and validation loss shown those two losses both converge after training, from 18 to nearly 8 and also will converge when increasing the number of epoches.

## II - PCA
In this section, the PCA is utilize from the `sklearn` framework. PCA is used to lower the dimension of the images. Specifically, an image, after pre-processed, is decomposed to a vector of size 512. The images are then fed to a simple fully connected networks with 3 Linear layers to classify 2 classes: `notumor` and `glioma`. 

However, the performance is not as expected. The results plot shows that both training loss and validation loss are fluctuate. The reason for this is that when decomposing an image (which is originally a 2D array) to a 1D array will drop the spatial patterns of the images and then lower the performance of the classifier. 