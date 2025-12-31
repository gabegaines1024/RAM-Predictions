import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

df = pd.read_csv('data/memory.csv')

# Drop rows with missing speed or modules (required for parsing)
df = df.dropna(subset=['speed', 'modules'])

# Parse the 'speed' column to extract DDR generation and MHz
# Use pd.to_numeric with errors='coerce' to handle any parsing issues
speed_split = df['speed'].str.split(',', expand=True)
df['ddr_gen'] = pd.to_numeric(speed_split[0], errors='coerce')
df['speed_mhz'] = pd.to_numeric(speed_split[1], errors='coerce')

# Parse 'modules' to get module count and size
modules_split = df['modules'].str.split(',', expand=True)
df['module_count'] = pd.to_numeric(modules_split[0], errors='coerce')
df['module_size'] = pd.to_numeric(modules_split[1], errors='coerce')

# Drop any rows where parsing failed
df = df.dropna(subset=['ddr_gen', 'speed_mhz', 'module_count', 'module_size'])

numeric_cols = ['price', 'price_per_gb', 'first_word_latency', 'cas_latency', 'speed_mhz']

# =============================================================================
# 1. HISTOGRAMS WITH LOG SCALE - Best for right-skewed price data
# =============================================================================
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Distribution of Key Features (Log Scale for Skewed Data)', fontsize=14, fontweight='bold')

# Price - log scale histogram
axs[0, 0].hist(df['price'].dropna(), bins=50, edgecolor='black', alpha=0.7)
axs[0, 0].set_xscale('log')
axs[0, 0].set_xlabel('Price ($) - Log Scale')
axs[0, 0].set_ylabel('Frequency')
axs[0, 0].set_title('Price Distribution')

# Price per GB - log scale
axs[0, 1].hist(df['price_per_gb'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='orange')
axs[0, 1].set_xscale('log')
axs[0, 1].set_xlabel('Price per GB ($) - Log Scale')
axs[0, 1].set_ylabel('Frequency')
axs[0, 1].set_title('Price per GB Distribution')

# CAS Latency - normal scale (typically not skewed)
axs[1, 0].hist(df['cas_latency'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='green')
axs[1, 0].set_xlabel('CAS Latency')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].set_title('CAS Latency Distribution')

# Speed MHz - normal scale
axs[1, 1].hist(df['speed_mhz'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='purple')
axs[1, 1].set_xlabel('Speed (MHz)')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].set_title('Memory Speed Distribution')

plt.tight_layout()
plt.show()

# =============================================================================
# 2. BOX PLOTS - Best for detecting outliers and comparing distributions
# =============================================================================
fig, axs = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Box Plots - Outlier Detection', fontsize=14, fontweight='bold')

# Price by DDR Generation
df.boxplot(column='price', by='ddr_gen', ax=axs[0])
axs[0].set_xlabel('DDR Generation')
axs[0].set_ylabel('Price ($)')
axs[0].set_title('Price by DDR Generation')

# Price per GB by DDR Generation
df.boxplot(column='price_per_gb', by='ddr_gen', ax=axs[1])
axs[1].set_xlabel('DDR Generation')
axs[1].set_ylabel('Price per GB ($)')
axs[1].set_title('Price/GB by DDR Generation')

# CAS Latency by DDR Generation
df.boxplot(column='cas_latency', by='ddr_gen', ax=axs[2])
axs[2].set_xlabel('DDR Generation')
axs[2].set_ylabel('CAS Latency')
axs[2].set_title('CAS Latency by DDR Generation')

plt.suptitle('')  # Remove auto-generated title
plt.tight_layout()
plt.show()

# =============================================================================
# 3. SCATTER PLOTS - Best for relationships (log scale for skewed targets)
# =============================================================================
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Feature vs Price Relationships', fontsize=14, fontweight='bold')

# Speed vs Price (log-log for both skewed)
axs[0, 0].scatter(df['speed_mhz'], df['price'], alpha=0.3, s=10)
axs[0, 0].set_yscale('log')
axs[0, 0].set_xlabel('Speed (MHz)')
axs[0, 0].set_ylabel('Price ($) - Log Scale')
axs[0, 0].set_title('Speed vs Price')

# CAS Latency vs Price
axs[0, 1].scatter(df['cas_latency'], df['price'], alpha=0.3, s=10, color='orange')
axs[0, 1].set_yscale('log')
axs[0, 1].set_xlabel('CAS Latency')
axs[0, 1].set_ylabel('Price ($) - Log Scale')
axs[0, 1].set_title('CAS Latency vs Price')

# Module Size vs Price
axs[1, 0].scatter(df['module_size'], df['price'], alpha=0.3, s=10, color='green')
axs[1, 0].set_yscale('log')
axs[1, 0].set_xlabel('Module Size (GB)')
axs[1, 0].set_ylabel('Price ($) - Log Scale')
axs[1, 0].set_title('Capacity vs Price')

# First Word Latency vs Price
axs[1, 1].scatter(df['first_word_latency'], df['price'], alpha=0.3, s=10, color='red')
axs[1, 1].set_yscale('log')
axs[1, 1].set_xlabel('First Word Latency (ns)')
axs[1, 1].set_ylabel('Price ($) - Log Scale')
axs[1, 1].set_title('Latency vs Price')

plt.tight_layout()
plt.show()

# =============================================================================
# 4. CORRELATION HEATMAP - Best for seeing all relationships at once
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

corr_cols = ['price', 'price_per_gb', 'speed_mhz', 'cas_latency', 'first_word_latency', 'module_size', 'ddr_gen']
corr_matrix = df[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, linewidths=0.5, fmt='.2f', ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =============================================================================
# 5. BAR CHARTS - Best for categorical data
# =============================================================================
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Categorical Feature Analysis', fontsize=14, fontweight='bold')

# Average price by DDR generation
ddr_prices = df.groupby('ddr_gen')['price'].mean().sort_index()
axs[0].bar(ddr_prices.index.astype(str), ddr_prices.values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'])
axs[0].set_xlabel('DDR Generation')
axs[0].set_ylabel('Average Price ($)')
axs[0].set_title('Average Price by DDR Generation')
for i, v in enumerate(ddr_prices.values):
    axs[0].text(i, v + 2, f'${v:.0f}', ha='center', fontweight='bold')

# Count by DDR generation
ddr_counts = df['ddr_gen'].value_counts().sort_index()
axs[1].bar(ddr_counts.index.astype(str), ddr_counts.values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'])
axs[1].set_xlabel('DDR Generation')
axs[1].set_ylabel('Number of Products')
axs[1].set_title('Product Count by DDR Generation')
for i, v in enumerate(ddr_counts.values):
    axs[1].text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================================================
# 6. VIOLIN PLOTS - Best for distribution shape comparison
# =============================================================================
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Price Distribution by DDR Generation', fontsize=14, fontweight='bold')

# Violin plot for price
sns.violinplot(data=df, x='ddr_gen', y='price', ax=axs[0], palette='Set2')
axs[0].set_yscale('log')
axs[0].set_xlabel('DDR Generation')
axs[0].set_ylabel('Price ($) - Log Scale')
axs[0].set_title('Price Distribution (Violin)')

# Violin plot for price per GB
sns.violinplot(data=df, x='ddr_gen', y='price_per_gb', ax=axs[1], palette='Set2')
axs[1].set_xlabel('DDR Generation')
axs[1].set_ylabel('Price per GB ($)')
axs[1].set_title('Price per GB Distribution (Violin)')

plt.tight_layout()
plt.show()

print("✅ All visualizations complete!")
print(f"\n📊 Dataset Summary:")
print(f"   Total records: {len(df):,}")
print(f"   DDR Generations: {sorted(df['ddr_gen'].unique())}")
print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
print(f"   Median price: ${df['price'].median():.2f}")