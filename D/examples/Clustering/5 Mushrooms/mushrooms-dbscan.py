import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load Data
df = pd.read_csv('mushrooms.csv')

# Since the mushroom data is categorical (text), we encode it into integers
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

# Selecting two features for 2D visualization: Cap Shape and Cap Color
X = df_encoded[['cap-shape', 'cap-color']].values

# 2. Scale the data
# Standardization is essential for DBSCAN to handle distances correctly
X_scaled = StandardScaler().fit_transform(X)

# 3. Initialize and Train DBSCAN
# eps: The neighborhood distance
# min_samples: Minimum points required to form a dense region
# We use a higher min_samples because categorical data often has many overlapping points
dbscan = DBSCAN(eps=0.3, min_samples=15)
clusters = dbscan.fit_predict(X_scaled)

# 4. Visualize the Results
plt.figure(figsize=(10, 6))
# Using alpha=0.1 because many mushrooms share the exact same shape and color
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=50, edgecolors='black', alpha=0.1)
plt.title("DBSCAN Clustering: Mushroom Dataset (Cap Shape vs Cap Color)")
plt.xlabel("Scaled Cap Shape")
plt.ylabel("Scaled Cap Color")
plt.colorbar(scatter, label='Cluster Label (-1 = Noise)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 5. Identifying Results
n_noise = list(clusters).count(-1)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

print(f"Number of clusters detected: {n_clusters}")
print(f"Number of noise points (outliers) detected: {n_noise}")