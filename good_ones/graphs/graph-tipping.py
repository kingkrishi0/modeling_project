import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

# Data
connection_survival = [0.076, 0.746, 0.782, 0.763, 0.620, 0.546, 0.465, 0.347, 0.331, 0.215, 0.241, 0.239, 0.233, 0.218]
ratio = [3, 4, 6, 7, 7.2, 7.3, 7.4, 7.49, 7.5, 8, 10, 12, 15, 20]

# Convert to numpy arrays for easier manipulation
x = np.array(ratio)
y = np.array(connection_survival)

# Create the main visualization
plt.figure(figsize=(12, 8))

# Main plot with enhanced styling
plt.subplot(2, 2, (1, 2))
plt.plot(x, y, 'o-', color='#2E86AB', linewidth=2.5, markersize=8, 
         markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=1.5,
         label='Connection Survival')

# Add trend line
z = np.polyfit(x, y, 2)  # 2nd degree polynomial fit
p = np.poly1d(z)
x_smooth = np.linspace(x.min(), x.max(), 100)
plt.plot(x_smooth, p(x_smooth), '--', color='#F18F01', linewidth=2, alpha=0.8, label='Trend Line')

# Identify the steepest drop point
differences = np.diff(y)
steepest_drop_idx = np.argmin(differences)
plt.axvline(x=x[steepest_drop_idx], color='red', linestyle=':', alpha=0.7, 
           label=f'Steepest Drop at Ratio {x[steepest_drop_idx]}')

# Styling
plt.xlabel('Ratio', fontsize=12, fontweight='bold')
plt.ylabel('Connection Survival', fontsize=12, fontweight='bold')
plt.title('Connection Survival vs Ratio Analysis', fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
plt.legend(framealpha=0.9, fontsize=10)

# Add text annotation for the critical point
critical_ratio = x[steepest_drop_idx]
critical_survival = y[steepest_drop_idx]
plt.annotate(f'Critical Point\n({critical_ratio}, {critical_survival:.3f})', 
            xy=(critical_ratio, critical_survival), xytext=(critical_ratio+2, critical_survival+0.1),
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
            fontsize=9)

# Statistical Analysis
print("=== CONNECTION SURVIVAL ANALYSIS ===\n")

# Basic statistics
print("DESCRIPTIVE STATISTICS:")
print(f"Mean Connection Survival: {np.mean(y):.4f}")
print(f"Std Deviation: {np.std(y):.4f}")
print(f"Min Survival: {np.min(y):.4f} at Ratio {x[np.argmin(y)]}")
print(f"Max Survival: {np.max(y):.4f} at Ratio {x[np.argmax(y)]}")
print(f"Range: {np.max(y) - np.min(y):.4f}")

# Correlation analysis
pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)

print(f"\nCORRELATION ANALYSIS:")
print(f"Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

# Significance interpretation
alpha = 0.05
if pearson_p < alpha:
    print(f"✓ Pearson correlation is statistically significant (p < {alpha})")
else:
    print(f"✗ Pearson correlation is not statistically significant (p ≥ {alpha})")

if spearman_p < alpha:
    print(f"✓ Spearman correlation is statistically significant (p < {alpha})")
else:
    print(f"✗ Spearman correlation is not statistically significant (p ≥ {alpha})")

# Linear regression analysis
slope, intercept, r_value, p_value_reg, std_err = stats.linregress(x, y)
print(f"\nLINEAR REGRESSION:")
print(f"Slope: {slope:.6f}")
print(f"Intercept: {intercept:.6f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"P-value: {p_value_reg:.4f}")
print(f"Standard Error: {std_err:.6f}")

# Identify critical transition points
print(f"\nCRITICAL TRANSITION ANALYSIS:")
print(f"Steepest decline occurs between Ratio {x[steepest_drop_idx]:.2f} and {x[steepest_drop_idx+1]:.2f}")
print(f"Survival drops by {abs(differences[steepest_drop_idx]):.3f} units")
print(f"This represents a {abs(differences[steepest_drop_idx])/y[steepest_drop_idx]*100:.1f}% decrease")

# Find where survival drops below certain thresholds
thresholds = [0.5, 0.3, 0.25]
for threshold in thresholds:
    below_threshold = x[y < threshold]
    if len(below_threshold) > 0:
        print(f"Connection survival drops below {threshold} at Ratio ≥ {below_threshold[0]}")

# Subplot 2: Residuals plot
plt.subplot(2, 2, 3)
predicted = slope * x + intercept
residuals = y - predicted
plt.scatter(x, residuals, color='#A23B72', alpha=0.7, s=60)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Ratio', fontweight='bold')
plt.ylabel('Residuals', fontweight='bold')
plt.title('Residuals Plot', fontweight='bold')
plt.grid(True, alpha=0.3)

# Subplot 3: Distribution of survival values
plt.subplot(2, 2, 4)
plt.hist(y, bins=8, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1)
plt.xlabel('Connection Survival', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Distribution of Survival Values', fontweight='bold')
plt.grid(True, alpha=0.3)

# Add mean line
plt.axvline(np.mean(y), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y):.3f}')
plt.legend()

plt.tight_layout()
plt.show()

# Create correlation matrix visualization
print(f"\nCORRELATION MATRIX:")
data_df = pd.DataFrame({'Ratio': x, 'Connection_Survival': y})
correlation_matrix = data_df.corr()
print(correlation_matrix)

# Additional analysis: Change rate analysis
print(f"\nCHANGE RATE ANALYSIS:")
rate_of_change = np.diff(y) / np.diff(x)
print("Rate of change between consecutive points:")
for i in range(len(rate_of_change)):
    print(f"  Ratio {x[i]:.2f} to {x[i+1]:.2f}: {rate_of_change[i]:.4f}")

# Find the most dramatic change
max_change_idx = np.argmax(np.abs(rate_of_change))
print(f"\nMost dramatic change: {rate_of_change[max_change_idx]:.4f}")
print(f"Occurs between Ratio {x[max_change_idx]:.2f} and {x[max_change_idx+1]:.2f}")

# Summary insights
print(f"\n=== KEY INSIGHTS ===")
print(f"1. Peak survival occurs at Ratio {x[np.argmax(y)]:.1f} with {np.max(y):.3f} survival rate")
print(f"2. Survival shows {pearson_corr:.3f} correlation with ratio (moderate {'negative' if pearson_corr < 0 else 'positive'})")
print(f"3. Critical drop-off begins around Ratio {x[steepest_drop_idx]:.2f}")
print(f"4. The relationship explains {r_value**2*100:.1f}% of the variance (R²)")

if pearson_p < 0.001:
    significance = "highly significant (p < 0.001)"
elif pearson_p < 0.01:
    significance = "very significant (p < 0.01)"
elif pearson_p < 0.05:
    significance = "significant (p < 0.05)"
else:
    significance = "not significant (p ≥ 0.05)"

print(f"5. The correlation is {significance}")