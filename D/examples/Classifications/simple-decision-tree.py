import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

'''
Predict whether a person will play tennis based on the weather
'''

# 1. Create a simple dataset
data = {
    'Outlook':  [0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2], # 0:Sunny, 1:Overcast, 2:Rain
    'Humidity': [0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1, 0, 1], # 0:High, 1:Normal, 2:Low
    'Wind':     [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],     # 0:Weak, 1:Strong
    'Play':     [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]      # 0:No, 1:Yes
}

df = pd.DataFrame(data)
print("df")
print(df)
X = df[['Outlook', 'Humidity', 'Wind']] # Features
y = df['Play']                         # Target

# 2. Split into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and Train the Decision Tree
# We use 'entropy' to measure the quality of the split
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train, y_train)

# 4. Predict and Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

# 5. Visualize the Tree
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=['Outlook', 'Humidity', 'Wind'], class_names=['No', 'Yes'], filled=True)
plt.show()

