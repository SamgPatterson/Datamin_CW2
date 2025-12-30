from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Features: [Cough (0: No, 1: Yes)]
# Classes: 0: Healthy, 1: Cold/Flu
X = [[0], [1], [1], [1], [1], [0]]
y = [0, 1, 1, 1, 1, 0]

clf_med = DecisionTreeClassifier(max_depth=3)
clf_med.fit(X, y)

prediction = clf_med.predict([[0]])
diagnosis = {0: "Healthy", 1: "Cold/Flu"}
print(f"The predicted diagnosis is: {diagnosis[prediction[0]]}")

# Visualize the tree
plt.figure(figsize=(10,6))
plot_tree(clf_med, feature_names=['Cough'], 
          class_names=['Healthy', 'Cold/Flu'], filled=True)
plt.show()