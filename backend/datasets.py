import numpy as np
from sklearn.datasets import make_circles, make_blobs, make_classification


def generate_dataset(dataset_type: str, noise: float = 0.1, n_samples: int = 200):
    if dataset_type == 'Circle':
        X, y = make_circles(n_samples=n_samples, noise=noise * 0.5, factor=0.5, random_state=42)
    elif dataset_type == 'Gaussian':
        X, y = make_blobs(n_samples=n_samples, centers=2, random_state=42)
        X = (X - X.mean(0)) / X.std(0)
    elif dataset_type == 'XOR':
        X, _ = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                   n_informative=2, random_state=42, n_clusters_per_class=1)
        mask = np.logical_or(
            np.logical_and(X[:, 0] > 0, X[:, 1] > 0),
            np.logical_and(X[:, 0] < 0, X[:, 1] < 0)
        )
        y = mask.astype(int)
        if noise > 0:
            flip_idx = np.random.RandomState(42).choice(n_samples, int(n_samples * noise * 0.05), replace=False)
            y[flip_idx] = 1 - y[flip_idx]
    elif dataset_type == 'Spiral':
        n_per_class = n_samples // 2
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples, dtype=int)
        rng = np.random.RandomState(42)
        for i in range(2):
            ix = range(n_per_class * i, n_per_class * (i + 1))
            r = np.linspace(0.0, 1, n_per_class)
            t = np.linspace(i * 2, (i + 1) * 2, n_per_class) + rng.randn(n_per_class) * noise * 0.1
            X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
            y[ix] = i
    else:
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)

    return X.tolist(), y.tolist()
