import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP: Path to your file on the C drive
file_path = r'C:\Practical data anylitcs and data mining\CW_2\ecoli.arff'

# 2. DEFINITION: Manually define the column names [cite: 10-17]
column_names = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]

# 3. LOAD DATA: configuration to handle ARFF metadata and comments
try:
    # comment='%' tells pandas to ignore lines starting with % [cite: 1-8]
    # skiprows=37 skips the metadata header to reach the data
    df = pd.read_csv(
        file_path, 
        names=column_names, 
        comment='%', 
        skiprows=37, 
        header=None, 
        sep=','
    )
    
    print("Success! Dataset loaded correctly.")
    print(f"Dataset Shape: {df.shape}") # Should show (336, 8) [cite: 8, 9]
    print(df.head()) 
    
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}. Please check the folder name.")
except Exception as e:
    print(f"An error occurred: {e}")

# 4. VISUALIZATION: Generating the required graphs
sns.set(style="whitegrid")

# Graph 1: Class Distribution [cite: 21-23]
plt.figure(figsize=(10, 6))
sns.countplot(x="class", data=df, palette="viridis", order=df["class"].value_counts().index)
plt.title("E.coli Dataset: Class Label Distribution", fontsize=14)
plt.xlabel("Localization Site (Class)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.savefig("ecoli_class_distribution.png")
plt.show()

