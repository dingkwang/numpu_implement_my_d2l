
"""
1. Randomly initialize k number of centroids points. 
2. Calculate the distance from each sample points(N) to k centroids points. 
3. Re-assign the label of the the N sample points based on their nearest. 
4. Update the centroids by the new group of labels. 
5. Loop step 2 to 4 until (max_iter, or tolerance). 

"""

import numpy as np


def k_means(X, k, max_iters=100, tol=1e-4):
    """
    Implement k-means clustering algorithm.

    Parameters:
    X : ndarray
        Input data, shape (n_samples, n_features)
    k : int
        Number of clusters
    max_iters : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance to declare convergence

    Returns:
    centroids : ndarray
        Final centroids, shape (k, n_features)
    labels : ndarray
        Labels of each point, shape (n_samples,)
    """

    # Randomly initialize centroids
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)] # (n, 2)

    for _ in range(max_iters):
        print(_)
        # Assign labels based on closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))

        labels = np.argmin(distances, axis=0)

        # Compute new centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence (if centroids do not change)
        if np.all(np.abs(new_centroids - centroids) < tol):
            print(min(np.abs(new_centroids - centroids)))
            break

        centroids = new_centroids

    return centroids, labels

# Example usage
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0], [1, 2.3], [1, 4.5], [1, 0.5], [10, 2.2], [10, 4.3], [10, 0.5]])
centroids, labels = k_means(X, 2)
print("Centroids:\n", centroids)
print("Labels:\n", labels)

"""
DBSCAN requires the min distance 
密度直达，密度可达，密度相连，非密度相连。
1，寻找核心点形成临时聚类簇。
2，合并临时聚类簇得到聚类簇。
https://zhuanlan.zhihu.com/p/336501183

"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        # Compute pairwise distances
        nbrs = NearestNeighbors(radius=self.eps).fit(X)
        distances, indices = nbrs.radius_neighbors(X)

        # Find core points
        core_points = np.array([len(neighbors) >= self.min_samples for neighbors in indices])

        # Initialize labels to -1 (unclassified)
        labels = np.full(X.shape[0], -1)
        cluster_id = 0

        for i in range(X.shape[0]):
            # Skip if not a core point or already classified
            if not core_points[i] or labels[i] != -1:
                continue

            # Start a new cluster
            labels[i] = cluster_id

            # Process neighbors for density-connected points
            seeds = set(indices[i])
            while seeds:
                current_point = seeds.pop()

                # Only process if it's a core point
                if core_points[current_point]:
                    for neighbor in indices[current_point]:
                        if labels[neighbor] == -1:
                            seeds.add(neighbor)
                            labels[neighbor] = cluster_id

            cluster_id += 1

        self.labels_ = labels

    def get_labels(self):
        return self.labels_

# Example usage
# X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
# dbscan = DBSCAN(eps=3, min_samples=2)
# dbscan.fit(X)
# print(dbscan.get_labels())
