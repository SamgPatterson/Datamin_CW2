import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np

# 1. Load Data
iris = load_iris()
X = iris.data[:, 2:4] # Petal Length and Petal Width

# 2. Initialize and Train
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X)

# 3. Predict on NEW data points
# Let's define two mystery flowers: 
# Flower A: Small (Length 1.5, Width 0.3)
# Flower B: Large (Length 5.5, Width 2.1)
new_flowers = np.array([[1.5, 0.3], 
                        [5.5, 2.1]])

new_predictions = kmeans.predict(new_flowers)

# 4. Print results
for i, pred in enumerate(new_predictions):
    print(f"Flower {i+1} (Measurements: {new_flowers[i]}) belongs to Cluster {pred}")

# 5. Visualizing everything
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='viridis', alpha=0.5, label='Training Data')

# Plot the Centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

# Plot the NEW predictions as bright stars
plt.scatter(new_flowers[:, 0], new_flowers[:, 1], c='orange', s=300, marker='*', edgecolors='black', label='New Predictions')

plt.title("K-Means: Training Data vs. New Predictions")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.show()