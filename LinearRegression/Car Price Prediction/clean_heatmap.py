import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set figure size for better readability
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load the data
df = pd.read_csv('CarPrice_Assignment.csv')
print(f"Dataset shape: {df.shape}")

# Data preprocessing
# Select only numeric columns for correlation analysis
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {numeric_columns}")

# Calculate correlation matrix
corr = df[numeric_columns].corr()
print(f"Correlation matrix shape: {corr.shape}")

# Original code (cluttered)
print("\nOriginal approach (cluttered):")
top_corr = corr[abs(corr) > 0.5].index
print(f"Number of features with correlation > 0.5: {len(top_corr)}")

# Create the original heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_corr].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Original Heatmap (Cluttered)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# IMPROVED VERSION 1: Focus on price correlations only
print("\nImproved Version 1: Price-focused correlations")

# Get correlations with price only
price_corr = corr['price'].sort_values(ascending=False)
print(f"\nTop 10 correlations with price:")
print(price_corr.head(10))

# Select top 10 features most correlated with price
top_price_features = price_corr.head(11).index.tolist()  # Include price itself

# Create clean heatmap
plt.figure(figsize=(12, 10))
price_corr_matrix = df[top_price_features].corr()

# Create mask for upper triangle to show only lower triangle
mask = np.triu(np.ones_like(price_corr_matrix, dtype=bool))

# Create the heatmap
sns.heatmap(price_corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdYlBu_r', 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8},
            linewidths=0.5,
            annot_kws={'size': 9})

plt.title('Price Correlation Heatmap (Top 10 Features)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# IMPROVED VERSION 2: Hierarchical clustering heatmap
print("\nImproved Version 2: Hierarchical clustering heatmap")

# Select features with moderate to high correlation (|correlation| > 0.3)
moderate_corr = corr[abs(corr) > 0.3].index
print(f"Features with |correlation| > 0.3: {len(moderate_corr)}")

# Create correlation matrix for these features
moderate_corr_matrix = df[moderate_corr].corr()

# Create the heatmap with clustering
plt.figure(figsize=(14, 12))

# Use clustermap for hierarchical clustering
g = sns.clustermap(moderate_corr_matrix,
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'shrink': 0.8},
                   linewidths=0.5,
                   annot_kws={'size': 8},
                   figsize=(14, 12))

plt.suptitle('Hierarchical Clustering Correlation Heatmap', 
             fontsize=16, fontweight='bold', y=0.95)
plt.show()

# IMPROVED VERSION 3: Clean correlation bar plot
print("\nImproved Version 3: Price correlation bar plot")

# Get absolute correlations with price (excluding price itself)
price_corr_abs = price_corr.drop('price').abs().sort_values(ascending=False)

# Create bar plot
plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(price_corr_abs)), price_corr_abs.values, 
               color=['red' if x < 0 else 'blue' for x in price_corr.drop('price')])

plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Absolute Correlation with Price', fontsize=12, fontweight='bold')
plt.title('Feature Correlations with Car Price', fontsize=14, fontweight='bold')
plt.xticks(range(len(price_corr_abs)), price_corr_abs.index, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add correlation values on bars
for i, (bar, value) in enumerate(zip(bars, price_corr_abs.values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# IMPROVED VERSION 4: Most practical approach - Focus on key features
print("\nImproved Version 4: Key features correlation matrix")

# Select the most important features for price prediction
key_features = ['price', 'enginesize', 'horsepower', 'curbweight', 
                'carlength', 'carwidth', 'wheelbase', 'citympg', 'highwaympg']

# Create correlation matrix for key features
key_corr_matrix = df[key_features].corr()

# Create clean heatmap
plt.figure(figsize=(10, 8))

# Create mask for upper triangle
mask = np.triu(np.ones_like(key_corr_matrix, dtype=bool))

sns.heatmap(key_corr_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8},
            linewidths=0.5,
            annot_kws={'size': 11})

plt.title('Key Features Correlation Matrix', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Print insights
print("\nKey Insights:")
print(f"1. Engine size has the highest correlation with price: {key_corr_matrix.loc['enginesize', 'price']:.3f}")
print(f"2. Horsepower correlation with price: {key_corr_matrix.loc['horsepower', 'price']:.3f}")
print(f"3. Fuel efficiency (city mpg) correlation with price: {key_corr_matrix.loc['citympg', 'price']:.3f}")
print(f"4. Car weight correlation with price: {key_corr_matrix.loc['curbweight', 'price']:.3f}") 