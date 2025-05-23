import numpy as np
from cvxopt import matrix, solvers

class SVMPolyDegree2:
    def __init__(self, C=1.0):
        self.C = C

    def polynomial_kernel(self, x, y):
        return (np.dot(x, y) + 1) ** 2

    def fit(self, X, y):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.polynomial_kernel(X[i], X[j])
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))
        h = matrix(np.hstack([np.zeros(n_samples), self.C * np.ones(n_samples)]))
        A = matrix(y.astype(np.double), (1, n_samples))
        b = matrix(0.0)
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])
        sv = alphas > 1e-5
        self.alpha = alphas[sv]
        self.support_vectors = X[sv]
        self.support_labels = y[sv]
        self.bias = np.mean([
            y_k - np.sum(self.alpha * self.support_labels *
                         [self.polynomial_kernel(x_k, x_sv) for x_sv in self.support_vectors])
            for x_k, y_k in zip(self.support_vectors, self.support_labels)
        ])

    def project(self, X):
        return np.array([
            np.sum(self.alpha * self.support_labels *
                   [self.polynomial_kernel(x, sv) for sv in self.support_vectors])
            + self.bias for x in X
        ])

    def predict(self, X):
        return np.sign(self.project(X))

class MultiClassSVM_OvR:
    def __init__(self, C=1.0):
        self.C = C
        self.models = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            binary_y = np.where(y == cls, 1, -1)
            clf = SVMPolyDegree2(C=self.C)
            clf.fit(X, binary_y)
            self.models[cls] = clf

    def predict(self, X):
        scores = np.column_stack([
            model.project(X) for model in self.models.values()
        ])
        best_class_indices = np.argmax(scores, axis=1)
        return np.array(self.classes)[best_class_indices] 