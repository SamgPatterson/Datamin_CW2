import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1. Setup the training data
data = {
    'Credit_Score': [400, 450, 500, 600, 700, 750, 650, 550],
    'Income': [30, 35, 50, 60, 80, 100, 45, 40],
    'Decision': [0, 0, 0, 1, 1, 1, 1, 0] # 0: Reject, 1: Approve
}
df = pd.DataFrame(data)
X = df[['Credit_Score', 'Income']]
y = df['Decision']

# 2. Train the model
clf_loan = DecisionTreeClassifier(max_depth=2)
clf_loan.fit(X, y)

# 3. Predict for a NEW applicant
# Let's say: Credit Score = 620, Income = 55 (thousands)
new_applicant = [[600, 15]]
prediction = clf_loan.predict(new_applicant)

# 4. Display the result in a readable way
status = "Approved" if prediction[0] == 1 else "Rejected"
print(f"Applicant Status: {status}")

# 5. Check the probability (Optional)
# This shows how confident the tree is (e.g., [0.2, 0.8] means 80% chance of Approval)
probability = clf_loan.predict_proba(new_applicant)
print(f"Probability (Reject vs Approve): {probability[0]}")