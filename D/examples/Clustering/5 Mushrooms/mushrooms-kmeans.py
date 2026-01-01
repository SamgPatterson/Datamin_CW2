import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# 1. Load Data
df = pd.read_csv('mushrooms.csv')

# Since K-Means requires numbers, we encode the categorical strings into integers
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

# Selecting two features for 2D visualization: Cap Shape and Cap Color
X = df_encoded[['cap-shape', 'cap-color']].values

# 2. Initialize and Train
# We use n_clusters=2 because mushrooms are primarily categorized as Edible or Poisonous
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans.fit(X)

# 3. Predict on NEW data points
# Let's define two mystery mushrooms using encoded values:
# Mushroom A: (Encoded Cap Shape 5, Cap Color 4)
# Mushroom B: (Encoded Cap Shape 2, Cap Color 8)
new_mushrooms = np.array([[5, 4], 
                          [2, 8]])

new_predictions = kmeans.predict(new_mushrooms)

# 4. Print results
print("Clustering Results for New Mushrooms:")
for i, pred in enumerate(new_predictions):
    print(f"Mushroom {i+1} (Encoded Features: {new_mushrooms[i]}) belongs to Cluster {pred}")

# 5. Visualizing everything
plt.figure(figsize=(10, 6))
# Using low alpha because there are many overlapping points in categorical data
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='viridis', alpha=0.1, label='Training Data')

# Plot the Centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')

# Plot the NEW predictions as bright stars
plt.scatter(new_mushrooms[:, 0], new_mushrooms[:, 1], c='orange', s=300, marker='*', edgecolors='black', label='New Predictions')

plt.title("K-Means: Mushroom Training Data (Cap Shape vs. Cap Color)")
plt.xlabel("Cap Shape (Encoded)")
plt.ylabel("Cap Color (Encoded)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()