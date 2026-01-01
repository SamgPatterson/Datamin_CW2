import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# 1. Load Data
iris = load_iris()
print(iris)
X = iris.data[:, 2:4] # We only take Petal Length and Petal Width for easy 2D plotting

# 2. Initialize and Train
# We choose n_clusters=3 because we know there are 3 species
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 3. Get Predictions (Cluster Labels)
y_kmeans = kmeans.predict(X)

# 4. Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Plot the Centroids (the "center" of each cluster)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')

plt.title("K-Means: Iris Petal Clusters")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()