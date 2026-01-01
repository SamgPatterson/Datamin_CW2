import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
import numpy as np

# 1. Load Data
cancer = load_breast_cancer()
# Using Feature 0 (mean radius) and Feature 1 (mean texture) for 2D plotting
X = cancer.data[:, :2] 

# 2. Initialize and Train
# We use n_clusters=2 since the dataset is naturally binary (Malignant/Benign)
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans.fit(X)

# 3. Predict on NEW data points
# Let's define two mystery samples: 
# Sample A: Small radius, smooth texture (likely Cluster 1)
# Sample B: Large radius, rough texture (likely Cluster 0)
new_samples = np.array([[10.5, 12.0], 
                        [22.0, 28.0]])

new_predictions = kmeans.predict(new_samples)

# 4. Print results
for i, pred in enumerate(new_predictions):
    print(f"Sample {i+1} (Measurements: {new_samples[i]}) belongs to Cluster {pred}")

# 5. Visualizing everything
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=30, cmap='coolwarm', alpha=0.6, label='Training Data')

# Plot the Centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='yellow', s=250, marker='X', edgecolors='black', label='Centroids')

# Plot the NEW predictions as bright stars
plt.scatter(new_samples[:, 0], new_samples[:, 1], c='lime', s=300, marker='*', edgecolors='black', label='New Predictions')

plt.title("K-Means Clustering: Breast Cancer Data (Radius vs Texture)")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()