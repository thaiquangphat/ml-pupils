from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class SklearnSVMPipeline:
    def __init__(self, degree=2, C=1.0, n_components=64, random_state=42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.model = SVC(kernel="poly", degree=degree, C=C, probability=True, random_state=random_state)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.model.fit(X_pca, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.model.predict(X_pca)

    def score(self, X, y):
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.model.score(X_pca, y) 