import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

# 1. Create Sample Data
# 'make_moons' creates two interleaving half-circles, which K-Means struggles with
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# 2. Scale the data (Crucial for distance-based density)
X_scaled = StandardScaler().fit_transform(X)

# 3. Initialize and Train DBSCAN
# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
# min_samples: The number of samples in a neighborhood for a point to be considered a core point
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# 4. Visualize the Results
plt.figure(figsize=(8, 6))
# Noise points are labeled as -1 by scikit-learn; we'll color them black
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("DBSCAN Clustering (Handles Non-Spherical Shapes)")
plt.xlabel("Scaled Feature 1")
plt.ylabel("Scaled Feature 2")
plt.colorbar(label='Cluster Label')
plt.show()

# 5. Identifying Noise
n_noise = list(clusters).count(-1)
print(f"Number of noise points detected: {n_noise}")
