from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class SklearnRandomForestPipeline:
    def __init__(self, n_estimators=100, n_components=64, random_state=42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

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