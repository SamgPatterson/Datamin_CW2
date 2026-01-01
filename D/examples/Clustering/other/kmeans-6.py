import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Data
iris = load_iris()
X = iris.data[:, 2:4] # Petal Length and Petal Width
y_true = iris.target  # Actual species (0=Setosa, 1=Versicolor, 2=Virginica)

# 2. Initialize and Train K-Means
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
y_clusters = kmeans.fit_predict(X) 

# 3. Generate Classification Report
# Note: The 'labels' in the report are now just 0, 1, and 2.
# Be careful: Cluster 1 might actually correspond to Species 0!
print("Classification Report (Raw Clusters):")
print(classification_report(y_true, y_clusters))

# 4. Visualize the Confusion Matrix
cm = confusion_matrix(y_true, y_clusters)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Cluster 0', 'Cluster 1', 'Cluster 2'], 
            yticklabels=['Setosa (0)', 'Versicolor (1)', 'Virginica (2)'])

plt.title('Confusion Matrix: Species vs. Raw Clusters')
plt.xlabel('K-Means Predicted Cluster')
plt.ylabel('Actual Species Label')
plt.show()