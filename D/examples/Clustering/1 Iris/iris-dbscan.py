import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 1. Load Data
iris = load_iris()
# Using Petal Length and Petal Width (indices 2 and 3)
X = iris.data[:, 2:4]

# 2. Scale the data
# DBSCAN relies on distance, so features must be on the same scale
X_scaled = StandardScaler().fit_transform(X)

# 3. Initialize and Train DBSCAN
# eps: The distance to search for neighbors
# min_samples: The minimum points required to form a "dense" region
# Note: For Iris, eps=0.2 works well to separate the distinct groups
dbscan = DBSCAN(eps=0.2, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# 4. Visualize the Results
plt.figure(figsize=(8, 6))
# Noise points are labeled as -1 by scikit-learn
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=50, edgecolors='black')
plt.title("DBSCAN Clustering: Iris Dataset (Petal Features)")
plt.xlabel("Scaled Petal Length")
plt.ylabel("Scaled Petal Width")
plt.colorbar(scatter, label='Cluster Label')
plt.show()

# 5. Identifying Noise
n_noise = list(clusters).count(-1)
print(f"Number of noise points detected: {n_noise}")