from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Sample Data (Text and Labels: 0 = Ham, 1 = Spam)
emails = [
    "Hey, are we still meeting for lunch today?",
    "URGENT: Your credit card account has been compromised. Click here.",
    "Congratulations! You've won a $1000 Walmart gift card. Claim now.",
    "Can you send me the notes from yesterday's lecture?",
    "Win a free vacation to the Bahamas by clicking this link.",
    "I'll be a bit late for the meeting, see you soon."
]
labels = [0, 1, 1, 0, 1, 0]

# 2. Split data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.3, random_state=42)

# 3. Convert text into a matrix of token counts (Vectorization)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

print(X_test_counts)

# 4. Initialize and Train the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# 5. Make Predictions
predictions = model.predict(X_test_counts)

# 6. Evaluate the Model
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print("\nDetailed Report:\n", classification_report(y_test, predictions))

# --- NEW: Confusion Matrix Visualization ---
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix: Spam Detection')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 7. Test with a completely new message
new_email = ["Hello Sam how are you?"]
new_email_counts = vectorizer.transform(new_email)
result = model.predict(new_email_counts)

print(f"New Email Prediction: {'Spam' if result[0] == 1 else 'Ham'}")

