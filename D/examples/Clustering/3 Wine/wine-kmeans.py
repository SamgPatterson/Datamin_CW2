import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
import numpy as np

# 1. Load Data
wine = load_wine()
# Using Feature 0 (Alcohol) and Feature 1 (Malic Acid) for 2D plotting
X = wine.data[:, :2] 

# 2. Initialize and Train
# The wine dataset naturally contains 3 different classes/cultivars
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X)

# 3. Predict on NEW data points
# Let's define two mystery wines: 
# Wine A: High alcohol, lower malic acid
# Wine B: Lower alcohol, higher malic acid
new_wines = np.array([[14.5, 1.5], 
                      [12.0, 4.0]])

new_predictions = kmeans.predict(new_wines)

# 4. Print results
for i, pred in enumerate(new_predictions):
    print(f"Wine {i+1} (Measurements: {new_wines[i]}) belongs to Cluster {pred}")

# 5. Visualizing everything
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='viridis', alpha=0.5, label='Training Data')

# Plot the Centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

# Plot the NEW predictions as bright stars
plt.scatter(new_wines[:, 0], new_wines[:, 1], c='orange', s=300, marker='*', edgecolors='black', label='New Predictions')

plt.title("K-Means Clustering: Wine Data (Alcohol vs Malic Acid)")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()