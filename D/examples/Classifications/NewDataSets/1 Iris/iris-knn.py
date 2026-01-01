import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling (CRITICAL for K-NN)
# K-NN uses distance math. If one feature is measured in meters and another in cm,
# the larger numbers will "bully" the smaller ones. Scaling levels the playing field.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Initialize and Train the K-NN (IBk) model
# n_neighbors=3 is a common starting point (K=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 5. Make Predictions
predictions = knn.predict(X_test)

# 6. Evaluate
print("Classification Report (K=3):\n", classification_report(y_test, predictions, target_names=iris.target_names))

# 7. Visualize the Confusion Matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('K-NN (IBk) Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()