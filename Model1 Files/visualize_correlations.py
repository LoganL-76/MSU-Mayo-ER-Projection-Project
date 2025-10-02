import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("weather+.csv")


# Select only numeric columns
numeric_df = df.select_dtypes(include=["number"])

# Compute correlations with 'count'
corr_with_count = numeric_df.corr()[["count"]].drop("count")

# Plot as heatmap
plt.figure(figsize=(4,6))
sns.heatmap(corr_with_count, annot=True, cmap="coolwarm", center=0)
plt.title("Patient Volume Correlation with various Features")
plt.tight_layout()
plt.show()
