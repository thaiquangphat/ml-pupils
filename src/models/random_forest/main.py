from random_forest.data_preprocessing import create_tf_datasets, process_image
from random_forest.feature_extraction import Combined_model, extract_features, apply_pca
from random_forest.rf_sklearn import SklearnRandomForestPipeline
from random_forest.rf_custom import RandomForest
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

    # Sklearn Random Forest
    rf_pipe = SklearnRandomForestPipeline()
    rf_pipe.fit(train_features, np.argmax(train_labels, axis=1))
    print("Sklearn RF Test Accuracy:", rf_pipe.score(test_features, np.argmax(test_labels, axis=1)))

    # Custom Random Forest
    train_features_pca, pca = apply_pca(train_features)
    test_features_pca = pca.transform(test_features)
    rf_custom = RandomForest()
    rf_custom.fit(train_features_pca, np.argmax(train_labels, axis=1))
    preds = rf_custom.predict(test_features_pca)
    acc = np.mean(preds == np.argmax(test_labels, axis=1))
    print("Custom RF Test Accuracy:", acc) 