import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data and calculate correlations
df = pd.read_csv('CarPrice_Assignment.csv')
corr = df.select_dtypes(include=[np.number]).corr()

# IMPROVED VERSION: Clean and focused heatmap
# Select top correlations with price (more focused than original 0.5 threshold)
price_corr = corr['price'].sort_values(ascending=False)
top_features = price_corr.head(8).index.tolist()  # Top 7 features + price

# Create clean heatmap
plt.figure(figsize=(10, 8))

# Create mask for upper triangle to avoid redundancy
mask = np.triu(np.ones_like(corr.loc[top_features, top_features], dtype=bool))

sns.heatmap(corr.loc[top_features, top_features], 
            mask=mask,
            annot=True, 
            cmap='RdYlBu_r', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8},
            linewidths=0.5,
            annot_kws={'size': 10})

plt.title('Price Correlation Heatmap (Top Features)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("Clean heatmap created! This version:")
print("1. Shows only the most important features")
print("2. Uses a mask to avoid redundant upper triangle")
print("3. Has better formatting and readability")
print("4. Focuses on price correlations specifically") 