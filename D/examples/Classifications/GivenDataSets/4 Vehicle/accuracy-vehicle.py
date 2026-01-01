import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv('vehicle.csv')

# 2. Preprocessing
X = df.drop('Class', axis=1)
y = df['Class']

# Encode the target labels (bus, saab, opel, van)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Feature Scaling (Crucial for KNN and improves convergence for others)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 3. Initialize Models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

# 4. Train and Evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

# 5. Tabulate Results
comparison_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
comparison_df = comparison_df.sort_values(by='Accuracy', ascending=False)
print(comparison_df)

# 6. Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=comparison_df, palette='viridis')
plt.title('Accuracy Comparison: Vehicle Classification')
plt.xlim(0, 1.1)
for i, v in enumerate(comparison_df['Accuracy']):
    plt.text(v + 0.01, i, f"{v:.2%}", va='center', fontweight='bold')
plt.tight_layout()
plt.show()