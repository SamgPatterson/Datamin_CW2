import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

# 1. Create the Nested Circles data
# factor=0.3 makes the inner circle very small compared to the outer ring
X, _ = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# 2. Run K-Means (k=2)
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
km_labels = kmeans.fit_predict(X_scaled)

# 3. Run DBSCAN
# We use the eps we found from the logic of the K-distance graph
dbscan = DBSCAN(eps=0.3, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

# 4. Plot Results Side-by-Side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# K-Means Plot
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=km_labels, cmap='Set1', s=30)
ax1.set_title("K-Means (Fails on Nested Shapes)")

# DBSCAN Plot
ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=db_labels, cmap='viridis', s=30)
ax2.set_title("DBSCAN (Succeeds on Density Path)")

plt.show()