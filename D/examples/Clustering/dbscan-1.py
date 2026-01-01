import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# 1. Prepare Data
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# 2. Calculate Nearest Neighbors
# We use min_samples = 5, so we look for the 5th nearest neighbor
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# 3. Sort distances (the 5th column represents the distance to the 5th neighbor)
distances = np.sort(distances[:, min_samples-1], axis=0)

# 4. Plot the K-Distance Graph
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title("K-Distance Graph (Finding Optimal EPS)")
plt.xlabel("Points Sorted by Distance")
plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance")
plt.axhline(y=0.25, color='r', linestyle='--') # The 'Elbow'
plt.grid(True)
plt.show()