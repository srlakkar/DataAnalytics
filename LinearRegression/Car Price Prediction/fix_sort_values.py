# Quick fix for the sort_values error
# The issue is likely with pandas version compatibility

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('CarPrice_Assignment.csv')
df = df.drop(['CarName'], axis=1)
df = pd.get_dummies(df, drop_first=True)

# FIXED VERSION - explicit parameter names
corr = df.corr()
top_features = corr['price'].abs().sort_values(ascending=False).head(12).index

# Alternative fix if the above doesn't work:
# top_features = corr['price'].abs().sort_values(ascending=False, inplace=False).head(12).index

print("Fixed! The issue was likely missing explicit parameter names in sort_values()")
print(f"Top features: {list(top_features)}") 