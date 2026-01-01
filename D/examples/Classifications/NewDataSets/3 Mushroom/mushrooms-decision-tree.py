import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Load Data
# (Assuming 'mushrooms.csv' is in your current directory)
df = pd.read_csv('mushrooms.csv')

# 2. Preprocessing
# Decision Trees in Scikit-Learn still need numbers instead of strings.
# LabelEncoder converts 'poisonous'/'edible' into 1/0.
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop('class', axis=1) # Features
y = df['class']              # Target (0=edible, 1=poisonous)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Decision Tree
# We use a small max_depth so we can actually read the final graph
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Visualize the Tree Logic
plt.figure(figsize=(20,10))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=['Edible', 'Poisonous'], 
          filled=True, 
          rounded=True,
          fontsize=12)
plt.title("Mushroom Classification Logic (Top 3 Levels)")
plt.show()

# 7. Generate the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 8. Plotting with Seaborn
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Edible', 'Poisonous'], 
            yticklabels=['Edible', 'Poisonous'])

plt.title('Decision Tree: Mushroom Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()