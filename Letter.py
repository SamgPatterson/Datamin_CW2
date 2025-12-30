import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP: Path to your file on the C drive
file_path = r'C:\Practical data anylitcs and data mining\CW_2\letter.arff'

# 2. DEFINITION: Manually define the column names
column_names = [
    "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", 
    "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", 
    "y-ege", "yegvx", "class"
]

# 3. LOAD DATA: configuration to handle ARFF metadata
try:
    # Most ARFF files have metadata that needs to be skipped
    # Skiprows should be adjusted based on the specific file header length
    df = pd.read_csv(
        file_path, 
        names=column_names, 
        comment='%', 
        skiprows=100, # Standard letter.arff often has ~100 metadata lines
        header=None, 
        sep=','
    )
    print("Success! Letter Recognition dataset loaded.")
    print(df.head()) 
except Exception as e:
    print(f"Error: {e}. You may need to adjust 'skiprows' to match your file's header.")

# 4. VISUALIZATION: Section II
sns.set(style="whitegrid")

# Graph 1: Class Distribution (A-Z)
plt.figure(figsize=(12, 6))
sns.countplot(x="class", data=df, order=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), palette="husl")
plt.title("Letter Recognition: Distribution of 26 Capital Letters", fontsize=14)
plt.savefig("letter_distribution.png")
plt.show()

# Graph 2: Feature Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = df.drop(columns=['class']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Image Features", fontsize=14)
plt.savefig("letter_heatmap.png")
plt.show()

# Graph 3: Box Plot of Box Heights by Letter
plt.figure(figsize=(15, 6))
sns.boxplot(x="class", y="high", data=df, order=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
plt.title("Variation of Box Height Across Different Letters", fontsize=14)
plt.savefig("letter_box_height.png")
plt.show()

# Graph 4: Scatter Plot (x-bar vs y-bar)
plt.figure(figsize=(10, 6))
# Sampling 1000 points for clarity in the scatter plot
sns.scatterplot(x="x-bar", y="y-bar", hue="class", data=df.sample(1000), alpha=0.5, legend=None)
plt.title("Mean X vs Mean Y Position of Pixels (Sample of 1000)", fontsize=14)
plt.savefig("letter_scatter_positions.png")
plt.show()

# Graph 5: Distribution of 'onpix' (Total pixels on)
plt.figure(figsize=(10, 6))
sns.histplot(df['onpix'], bins=16, kde=True, color='purple')
plt.title("Distribution of Total 'On' Pixels Across All Stimuli", fontsize=14)
plt.savefig("letter_onpix_distribution.png")
plt.show()