import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import re

# 1. Load Data
# Using 'latin-1' to handle special characters in SMS text
df = pd.read_csv('spam.csv', encoding='latin-1')

# Create numerical features for clustering
df['message_length'] = df['v2'].apply(len)
df['digit_count'] = df['v2'].apply(lambda x: len(re.findall(r'\d', x)))

# Feature matrix X
X = df[['message_length', 'digit_count']].values

# 2. Visualize the Dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram - SMS Spam Dataset (Length vs Digits)")
# 'ward' linkage minimizes the variance within clusters
linkage_matrix = linkage(X, method='ward')
# Using truncate_mode because the dataset has over 5,000 samples
dendrogram(linkage_matrix, truncate_mode='lastp', p=20) 

# Threshold line for 2 clusters (Ham vs Spam)
plt.axhline(y=4000, color='r', linestyle='--', label='2 Cluster Threshold') 
plt.legend()
plt.show()

# 3. Initialize and Train the Agglomerative Model
# We choose 2 clusters to match the nature of the dataset (Ham/Spam)
cluster_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = cluster_model.fit_predict(X)

# 4. Visualize the Final Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', edgecolors='black', alpha=0.5)
plt.title("Agglomerative Clustering Results on Spam (k=2)")
plt.xlabel("Message Length (Characters)")
plt.ylabel("Digit Count")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()