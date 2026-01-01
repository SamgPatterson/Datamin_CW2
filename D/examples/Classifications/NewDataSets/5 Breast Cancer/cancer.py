from sklearn.datasets import load_breast_cancer

# Load the dataset object (a "Bunch" object)
cancer = load_breast_cancer()

# 1. Print the full documentation of the dataset
print(cancer.DESCR)

# 2. See exactly what features are being measured
print("Features:", cancer.feature_names)

# 3. See what the numbers 0 and 1 represent in the target
print("Target Labels:", cancer.target_names)

# 4. Convert to a DataFrame for a quick statistical look
import pandas as pd
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
print(df.describe())