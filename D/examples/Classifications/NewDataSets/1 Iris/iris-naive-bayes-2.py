import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# 1. Load Data (Only 2 features for 2D plotting)
iris = load_iris()
X = iris.data[:, 2:4]  # Petal Length and Petal Width
y = iris.target

# 2. Train the Model
gnb = GaussianNB()
gnb.fit(X, y)

# 3. Create a mesh grid to plot boundaries
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 4. Predict across the entire mesh grid
Z = gnb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 5. Plotting
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis') # The colored regions
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis', s=50) # The actual flowers

plt.title("Naive Bayes Decision Boundaries (Petal Length vs Width)")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()