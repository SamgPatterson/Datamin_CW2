import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the dataset (Assuming you downloaded 'spam.csv' from Kaggle)
# Kaggle's version usually requires latin-1 encodingclea
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']] # Keep only label and text
df.columns = ['label', 'message']

# 2. Preprocessing
# Convert labels to binary: 0 for ham, 1 for spam
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# 4. Feature Extraction: "Bag of Words"
# This turns sentences into a matrix of word counts
cv = CountVectorizer(stop_words='english')
X_train_counts = cv.fit_transform(X_train)
X_test_counts = cv.transform(X_test)

# 5. Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# 6. Evaluate
predictions = model.predict(X_test_counts)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nDetailed Report:\n", classification_report(y_test, predictions))

# 7. Generate the Confusion Matrix data
cm = confusion_matrix(y_test, predictions)

# 8. Visualize using Seaborn
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])

plt.title('Spam Filter Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()