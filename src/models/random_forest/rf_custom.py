import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1] if self.n_features is None else self.n_features
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(set(y))
        if depth >= self.max_depth or num_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return {"leaf": leaf_value}
        feat_idxs = np.random.choice(n_features, self.n_features_, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            return {"leaf": self._most_common_label(y)}
        left_idxs = X[:, best_feat] < best_thresh
        right_idxs = X[:, best_feat] >= best_thresh
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return {"leaf": self._most_common_label(y)}
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return {"feature": best_feat, "threshold": best_thresh, "left": left, "right": right}

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for idx in feat_idxs:
            thresholds = np.unique(X[:, idx])
            for t in thresholds:
                gain = self._information_gain(y, X[:, idx], t)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thresh = t
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        left_mask = X_column < split_thresh
        right_mask = ~left_mask
        if sum(left_mask) == 0 or sum(right_mask) == 0:
            return 0
        p = len(y)
        entropy_before = self._entropy(y)
        entropy_after = (len(y[left_mask]) / p) * self._entropy(y[left_mask]) + \
                        (len(y[right_mask]) / p) * self._entropy(y[right_mask])
        return entropy_before - entropy_after

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _predict(self, x, tree):
        if "leaf" in tree:
            return tree["leaf"]
        if x[tree["feature"]] < tree["threshold"]:
            return self._predict(x, tree["left"])
        else:
            return self._predict(x, tree["right"])

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                n_features=int(np.sqrt(n_features)))
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds) 