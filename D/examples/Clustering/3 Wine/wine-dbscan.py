import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# 1. Load Data
wine = load_wine()
# Using Alcohol (index 0) and Malic Acid (index 1)
X = wine.data[:, :2]

# 2. Scale the data
# Since Alcohol and Malic Acid have different scales, standardization is required
X_scaled = StandardScaler().fit_transform(X)

# 3. Initialize and Train DBSCAN
# eps: The search radius around each point
# min_samples: The minimum number of points to form a cluster core
# For the wine dataset, eps=0.5 and min_samples=5 provide a clear grouping
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# 4. Visualize the Results
plt.figure(figsize=(8, 6))
# Noise points (outliers) are labeled as -1 and colored differently
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=50, edgecolors='black', alpha=0.8)
plt.title("DBSCAN Clustering: Wine Dataset (Alcohol vs Malic Acid)")
plt.xlabel("Scaled Alcohol")
plt.ylabel("Scaled Malic Acid")
plt.colorbar(scatter, label='Cluster Label (-1 = Noise)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 5. Identifying Results
n_noise = list(clusters).count(-1)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

print(f"Number of clusters detected: {n_clusters}")
print(f"Number of noise points (outliers) detected: {n_noise}")