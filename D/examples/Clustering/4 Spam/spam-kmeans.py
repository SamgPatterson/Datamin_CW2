import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import re

# 1. Load Data
df = pd.read_csv('spam.csv', encoding='latin-1')

# Create numerical features from the text
# Feature 1: Character length of the message
# Feature 2: Number of digits in the message (common in spam)
df['message_length'] = df['v2'].apply(len)
df['digit_count'] = df['v2'].apply(lambda x: len(re.findall(r'\d', x)))

# Select our feature matrix X
X = df[['message_length', 'digit_count']].values

# 2. Initialize and Train
# We use 2 clusters because the data is labeled as 'ham' or 'spam'
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans.fit(X)

# 3. Predict on NEW data points
# Message A: A short personal text like "Hi, how are you?"
# Message B: A longer promotional text like "WINNER! You won 1000 cash. Call 0800123456 now."
new_messages = np.array([[18, 0], 
                         [100, 15]])

new_predictions = kmeans.predict(new_messages)

# 4. Print results
print("Clustering Results for New Messages:")
for i, pred in enumerate(new_predictions):
    print(f"Message {i+1} (Length: {new_messages[i][0]}, Digits: {new_messages[i][1]}) belongs to Cluster {pred}")

# 5. Visualizing everything
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=30, cmap='coolwarm', alpha=0.4, label='Training Data')

# Plot the Centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='yellow', s=200, marker='X', edgecolors='black', label='Centroids')

# Plot the NEW predictions as bright stars
plt.scatter(new_messages[:, 0], new_messages[:, 1], c='lime', s=300, marker='*', edgecolors='black', label='New Predictions')

plt.title("K-Means Clustering: SMS Spam Data (Length vs. Digits)")
plt.xlabel("Message Length (Characters)")
plt.ylabel("Digit Count")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()