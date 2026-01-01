import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
df = pd.read_csv('ecoli.csv')

# Features and Target
X = df.drop('class', axis=1)
y = df['class']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Initialize and train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42) # No depth limit for performance, but limited for plot later
dt_model.fit(X_train, y_train)

# Predictions
y_pred = dt_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

# Get the labels present in the test set for the report
present_classes = np.unique(np.concatenate([y_test, y_pred]))
target_names = le.inverse_transform(present_classes)

report = classification_report(y_test, y_pred, labels=present_classes, target_names=target_names)

# Confusion Matrix plot
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('ecoli_dt_confusion_matrix.png')

# Decision Tree Plot (constrained depth for visibility)
plt.figure(figsize=(20, 10))
dt_viz = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_viz.fit(X_train, y_train)
plot_tree(dt_viz, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Path (Max Depth 4)")
plt.savefig('ecoli_decision_tree_viz.png')

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)