import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_breast_cancer
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load Breast Cancer Data
cancer = load_breast_cancer()
# Selecting Mean Radius (index 0) and Mean Texture (index 1) for 2D visualization
X = cancer.data[:, :2] 

# 2. Visualize the Dendrogram
# Note: Because the dataset has 569 samples, the dendrogram will be very dense.
plt.figure(figsize=(10, 7))
plt.title("Dendrogram - Breast Cancer Dataset (Radius vs Texture)")
# 'ward' linkage minimizes the variance within clusters
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=12) # Truncated for readability

# Threshold line where the tree splits into 2 clear clusters
plt.axhline(y=500, color='r', linestyle='--', label='2 Cluster Threshold') 
plt.legend()
plt.show()

# 3. Initialize and Train the Agglomerative Model
# We choose 2 clusters to match the Malignant/Benign nature of the data
cluster_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = cluster_model.fit_predict(X)

# 4. Visualize the Final Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', edgecolors='black', alpha=0.7)
plt.title("Agglomerative Clustering Results (k=2)")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()