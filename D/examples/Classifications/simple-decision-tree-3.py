from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Features: [Temperature (Celsius), Cough (0: No, 1: Yes)]
# Classes: 0: Healthy, 1: Cold, 2: Flu
X = [[36.5, 0], [37.2, 1], [38.5, 1], [39.0, 1], [36.8, 1], [37.0, 0]]
y = [0, 1, 2, 2, 1, 0]

clf_med = DecisionTreeClassifier(max_depth=3)
clf_med.fit(X, y)

# Test with a new patient: 38.2 degrees and coughing
prediction = clf_med.predict([[36.2, 0]])
diagnosis = {0: "Healthy", 1: "Cold", 2: "Flu"}
print(f"The predicted diagnosis is: {diagnosis[prediction[0]]}")

# Visualize the tree
plt.figure(figsize=(10,6))
plot_tree(clf_med, feature_names=['Temp', 'Cough'], 
          class_names=['Healthy', 'Cold', 'Flu'], filled=True)
plt.show()