# Quick fix for the deprecated sns.set() function
# Replace this line:
# sns.set(font_scale=1.0)

# With this line:
# sns.set_theme(font_scale=1.0)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('CarPrice_Assignment.csv')
df = df.drop(['CarName'], axis=1)
df = pd.get_dummies(df, drop_first=True)

# Select top features
corr = df.corr()
top_features = corr['price'].abs().sort_values(ascending=False).head(10).index
top_corr_matrix = df[top_features].corr()

# FIXED VERSION - using set_theme instead of set
plt.figure(figsize=(12, 6))
sns.set_theme(font_scale=1.0)  # This is the fix!
sns.heatmap(top_corr_matrix, annot=True, fmt=".2f", linewidths=0.5, square=True, cbar_kws={'shrink': .8})
plt.title('Correlation Heatmap of Top Features', fontsize=14)
plt.tight_layout()
plt.show()

print("Fixed! Changed sns.set() to sns.set_theme()") 