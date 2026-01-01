import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and Train the Model
# No complex tuning required for Naive Bayes!
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 4. Make Predictions
y_pred = gnb.predict(X_test)

# 5. Evaluate the Model
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Naive Bayes: Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()