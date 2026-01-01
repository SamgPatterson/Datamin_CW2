import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_wine
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load Wine Data
wine = load_wine()
# Selecting Alcohol (index 0) and Malic Acid (index 1) for 2D visualization
X = wine.data[:, :2] 

# 2. Visualize the Dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram - Wine Dataset (Alcohol vs Malic Acid)")
# 'ward' linkage minimizes the variance within clusters
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=15) # Truncated for clarity

# Threshold line where the tree splits into 3 clear clusters
plt.axhline(y=10, color='r', linestyle='--', label='3 Cluster Threshold') 
plt.legend()
plt.show()

# 3. Initialize and Train the Agglomerative Model
# We choose 3 clusters to match the 3 wine cultivars
cluster_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = cluster_model.fit_predict(X)

# 4. Visualize the Final Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='black', alpha=0.8)
plt.title("Agglomerative Clustering Results on Wine (k=3)")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()