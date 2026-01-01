import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import mode

# 1. Load Data
iris = load_iris()
X = iris.data[:, 2:4]  # Petal Length and Petal Width
y_true = iris.target   # Actual species labels

# 2. Initialize and Train K-Means
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
y_clusters = kmeans.fit_predict(X)

# 3. Label Mapping (Essential for K-Means Evaluation)
# We map each cluster ID to the most frequent actual species in that cluster
labels = np.zeros_like(y_clusters)
for i in range(3):
    mask = (y_clusters == i)
    # Match the cluster to the real target name
    labels[mask] = mode(y_true[mask], keepdims=True).mode[0]

# 4. Generate Classification Report
print("Classification Report:\n")
print(classification_report(y_true, labels, target_names=iris.target_names))

# 5. Visualize the Confusion Matrix
cm = confusion_matrix(y_true, labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)

plt.title('K-Means Performance: Confusion Matrix')
plt.xlabel('Predicted Species (from clusters)')
plt.ylabel('Actual Species')
plt.show()