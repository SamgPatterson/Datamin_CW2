import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load Mushroom Data
df = pd.read_csv('mushrooms.csv')

# Encode categorical data to numerical so the algorithm can process it
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

# Selecting Cap Shape and Cap Color for 2D visualization
X = df_encoded[['cap-shape', 'cap-color']].values

# 2. Visualize the Dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram - Mushroom Dataset (Cap Shape vs Cap Color)")
# 'ward' linkage minimizes the variance within clusters
linkage_matrix = linkage(X, method='ward')
# Truncating the dendrogram because the dataset has over 8,000 rows
dendrogram(linkage_matrix, truncate_mode='lastp', p=12) 

# Threshold line - chosen to split the data into 2 main clusters (Edible vs Poisonous)
plt.axhline(y=150, color='r', linestyle='--', label='2 Cluster Threshold') 
plt.legend()
plt.show()

# 3. Initialize and Train the Agglomerative Model
# We choose 2 clusters to represent the Edible and Poisonous categories
cluster_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = cluster_model.fit_predict(X)

# 4. Visualize the Final Clusters
plt.figure(figsize=(8, 6))
# Using a low alpha (0.1) because many mushrooms share the same encoded features
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='black', alpha=0.1)
plt.title("Agglomerative Clustering Results on Mushrooms (k=2)")
plt.xlabel("Cap Shape (Encoded)")
plt.ylabel("Cap Color (Encoded)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()