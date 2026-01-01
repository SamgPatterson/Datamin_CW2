import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Load Data
wine = load_wine()
X = wine.data
y = wine.target

# 2. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and Train the Model
# We set max_depth to keep the tree simple and avoid 'overfitting'
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 4. Make Predictions
y_pred = clf.predict(X_test)

# 5. Evaluate the Model
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=wine.target_names))

# 6. Visualize the Decision Tree logic
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)
plt.title("Decision Tree Logic for Wine Classification")
plt.show()

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title('Decision Tree: Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()