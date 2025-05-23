from svm_kernel_polydeg2.data_preprocessing import create_tf_datasets, process_image
from svm_kernel_polydeg2.feature_extraction import Combined_model, extract_features, apply_pca
from svm_kernel_polydeg2.svm_sklearn import SklearnSVMPipeline
from svm_kernel_polydeg2.svm_custom import MultiClassSVM_OvR
import numpy as np

# Example usage (paths and parameters may need adjustment)
if __name__ == "__main__":
    base_dir = 'brain-tumor-mri-dataset'  # Adjust as needed
    train_data, val_data, test_data = create_tf_datasets(base_dir)
    train_data = train_data.map(process_image)

    # Feature extraction
    fe_model = Combined_model()
    # fe_model.fit(train_data, epochs=1)  # Uncomment to train feature extractor if needed
    train_features, train_labels = extract_features(train_data, fe_model)
    test_features, test_labels = extract_features(test_data, fe_model)

    # Sklearn SVM
    svm_pipe = SklearnSVMPipeline()
    svm_pipe.fit(train_features, np.argmax(train_labels, axis=1))
    print("Sklearn SVM Test Accuracy:", svm_pipe.score(test_features, np.argmax(test_labels, axis=1)))

    # Custom SVM (One-vs-Rest)
    train_features_pca, pca = apply_pca(train_features)
    test_features_pca = pca.transform(test_features)
    svm_custom = MultiClassSVM_OvR()
    svm_custom.fit(train_features_pca, np.argmax(train_labels, axis=1))
    preds = svm_custom.predict(test_features_pca)
    acc = np.mean(preds == np.argmax(test_labels, axis=1))
    print("Custom SVM Test Accuracy:", acc) 