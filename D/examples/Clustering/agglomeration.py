import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Create Sample Data (points on a 2D plane)
X = np.array([[5,3], [10,15], [15,12], [24,10], [30,30], [85,70], [71,80], [60,78], [70,55], [80,91]])

# 2. Visualize the Dendrogram (The 'Family Tree' of the data)
# This helps us decide how many clusters to use
plt.figure(figsize=(10, 7))
plt.title("Dendrogram - Finding the Optimal Number of Clusters")
# 'ward' linkage minimizes the variance of clusters being merged
linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix)
plt.axhline(y=35, color='r', linestyle='--') # Example threshold line
plt.show()

# 3. Initialize and Train the Agglomerative Model
# Based on the dendrogram, we'll choose 2 clusters
cluster_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = cluster_model.fit_predict(X)

# 4. Visualize the Final Clusters
plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow')
plt.title("Agglomerative Clustering Results (k=2)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
