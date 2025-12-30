import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and Train the Random Forest
# n_estimators=100 means we are building a forest of 100 trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Make Predictions
y_pred = rf_model.predict(X_test)

# 5. Evaluate the Model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_test))

# 6. Check Feature Importance
# This tells us which physical traits of the flower were most important
feature_imp = pd.Series(rf_model.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
print("\nFeature Importances:\n", feature_imp)

# 7. Visualize Feature Importance
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title("Key Features for Flower Classification")
plt.show()


# 1. Generate the confusion matrix
# y_test = actual labels, y_pred = model predictions
cm = confusion_matrix(y_test, y_pred)

# 2. Create the Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)

# 3. Add Labels and Title
plt.title('Confusion Matrix Heatmap: Iris Classification')
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.show()


# Generate report as a dictionary
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Convert to DataFrame and remove 'accuracy' and 'support' for a cleaner plot
report_df = pd.DataFrame(report_dict).iloc[:-1, :3].T

plt.figure(figsize=(10, 5))
sns.heatmap(report_df, annot=True, cmap='RdYlGn', center=0.8)
plt.title('Classification Metrics Heatmap')
plt.show()