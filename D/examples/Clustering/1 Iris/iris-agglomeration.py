import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load Iris Data
iris = load_iris()
# Selecting Petal Length (index 2) and Petal Width (index 3) for 2D visualization
X = iris.data[:, 2:4] 

# 2. Visualize the Dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram - Iris Dataset (Petal Features)")
# 'ward' linkage minimizes the variance within clusters
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)

# We add a threshold line where the tree splits into 3 clear clusters
plt.axhline(y=10, color='r', linestyle='--', label='3 Cluster Threshold') 
plt.legend()
plt.show()

# 3. Initialize and Train the Agglomerative Model
# We choose 3 clusters to match the 3 species in the dataset
cluster_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = cluster_model.fit_predict(X)

# 4. Visualize the Final Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', edgecolors='black')
plt.title("Agglomerative Clustering Results (k=3)")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()