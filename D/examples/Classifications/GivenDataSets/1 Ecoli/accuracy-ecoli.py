import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('ecoli.csv')

# Features and Target
X = df.drop('class', axis=1)
y = df['class']

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data (using 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Models to compare
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

# Evaluate models using Cross-Validation (mean accuracy)
cv_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
    cv_results[name] = cv_scores.mean()

# Evaluate on test set
test_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    test_results[name] = model.score(X_test, y_test)

# Combine results into a DataFrame
comparison_df = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'CV Accuracy': list(cv_results.values()),
    'Test Accuracy': list(test_results.values())
}).sort_values(by='Test Accuracy', ascending=False)

# Save to CSV
comparison_df.to_csv('ecoli_accuracy_comparison.csv', index=False)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Test Accuracy', y='Model', data=comparison_df, palette='magma')
plt.title('Accuracy Comparison on Ecoli Dataset')
plt.xlim(0, 1.0)
for i, v in enumerate(comparison_df['Test Accuracy']):
    plt.text(v + 0.01, i, f"{v:.2%}", color='black', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('ecoli_accuracy_plot.png')

print(comparison_df)