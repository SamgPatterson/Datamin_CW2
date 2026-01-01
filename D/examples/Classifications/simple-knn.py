import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Sample Data: Predicting if a person buys a product based on Age and Salary
data = {
    'Age': [22, 25, 47, 52, 46, 56, 34, 33, 21, 28, 35, 39, 41, 48],
    'Salary': [20000, 25000, 70000, 90000, 60000, 100000, 40000, 35000, 18000, 30000, 50000, 55000, 62000, 85000],
    'Purchased': [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1] # 0: No, 1: Yes
}

df = pd.DataFrame(data)
X = df[['Age', 'Salary']]
y = df['Purchased']

# 2. Split into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_test)

# 3. FEATURE SCALING (Crucial for K-NN)
# This scales the data so Age and Salary have equal weight in distance calculations
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_test)

# 4. Initialize and Train the K-NN Classifier
# We'll look at the 3 nearest neighbors (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 5. Make Predictions
y_pred = knn.predict(X_test)

# 6. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Print a detailed report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Purples', fmt='g')
plt.xlabel('Predicted (Purchased)')
plt.ylabel('Actual (Purchased)')
plt.title('K-NN Confusion Matrix')
plt.show()

error_rate = []

# Check k values from 1 to 10
for i in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.plot(range(1, 11), error_rate, marker='o', color='blue')
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()