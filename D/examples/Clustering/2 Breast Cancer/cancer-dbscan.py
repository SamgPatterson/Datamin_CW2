import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# 1. Load Data
cancer = load_breast_cancer()
# Using Mean Radius and Mean Texture (indices 0 and 1)
X = cancer.data[:, :2]

# 2. Scale the data
# Crucial for DBSCAN! Without scaling, the feature with the larger 
# range would dominate the distance calculation.
X_scaled = StandardScaler().fit_transform(X)

# 3. Initialize and Train DBSCAN
# eps: The maximum distance between two samples to be considered neighbors
# min_samples: The number of samples in a neighborhood for a point to be a core point
# For this dataset, eps=0.4 provides a good balance between clusters and noise
dbscan = DBSCAN(eps=0.4, min_samples=10)
clusters = dbscan.fit_predict(X_scaled)

# 4. Visualize the Results
plt.figure(figsize=(8, 6))
# Points labeled -1 are considered 'Noise' (Outliers)
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='plasma', s=40, edgecolors='black', alpha=0.7)
plt.title("DBSCAN Clustering: Breast Cancer (Radius vs Texture)")
plt.xlabel("Scaled Mean Radius")
plt.ylabel("Scaled Mean Texture")
plt.colorbar(scatter, label='Cluster Label (-1 = Noise)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 5. Identifying Results
n_noise = list(clusters).count(-1)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

print(f"Number of clusters detected: {n_clusters}")
print(f"Number of noise points (outliers) detected: {n_noise}")