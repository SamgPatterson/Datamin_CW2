import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# 1. Generate synthetic data (3 distinct groups)
# In a real scenario, you would load your CSV here
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 2. Scale the data
# K-Means calculates distance, so we need features on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Initialize and Train the K-Means Model
# We tell the model to find 3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

# 4. Add the cluster labels back to our data
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['Cluster'] = clusters

# 5. Visualize the results
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, s=50, cmap='viridis')

# Plot the centroids (the "center" of each cluster)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-Means Clustering Results")
plt.xlabel("Scaled Feature 1")
plt.ylabel("Scaled Feature 2")
plt.show()
