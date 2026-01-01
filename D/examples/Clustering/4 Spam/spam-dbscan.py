import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import re

# 1. Load Data
# Using 'latin-1' encoding to handle special characters often found in SMS datasets
df = pd.read_csv('spam.csv', encoding='latin-1')

# Create numerical features from the text
# Feature 1: Total character length
# Feature 2: Count of digits (common in spam/promotional messages)
df['message_length'] = df['v2'].apply(len)
df['digit_count'] = df['v2'].apply(lambda x: len(re.findall(r'\d', x)))

# Selecting our feature matrix X
X = df[['message_length', 'digit_count']].values

# 2. Scale the data
# Standardizing ensures that length and digit count contribute equally to the distance
X_scaled = StandardScaler().fit_transform(X)

# 3. Initialize and Train DBSCAN
# eps: The search radius (smaller values mean higher density required)
# min_samples: Minimum points needed to form a "dense" cluster core
# For this larger dataset, we use a higher min_samples to ignore small groups of outliers
dbscan = DBSCAN(eps=0.3, min_samples=10)
clusters = dbscan.fit_predict(X_scaled)

# 4. Visualize the Results
plt.figure(figsize=(10, 6))
# Points labeled as -1 (Noise/Outliers) appear in the background color of the map
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=30, edgecolors='black', alpha=0.5)
plt.title("DBSCAN Clustering: SMS Spam Dataset (Length vs Digits)")
plt.xlabel("Scaled Message Length")
plt.ylabel("Scaled Digit Count")
plt.colorbar(scatter, label='Cluster Label (-1 = Noise)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 5. Identifying Results
n_noise = list(clusters).count(-1)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

print(f"Number of clusters detected: {n_clusters}")
print(f"Number of noise points (outliers) detected: {n_noise}")