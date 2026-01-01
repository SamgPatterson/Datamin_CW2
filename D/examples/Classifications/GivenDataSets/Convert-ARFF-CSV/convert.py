import arff
import pandas as pd

# Load the ARFF file
with open('vote.arff', 'r') as f:
    data = arff.load(f)

# Convert to Pandas DataFrame
# 'data['data']' contains the rows, 'data['attributes']' contains the column names
attributes = [attr[0] for attr in data['attributes']]
df = pd.DataFrame(data['data'], columns=attributes)

# Save as CSV
df.to_csv('vote.csv', index=False)