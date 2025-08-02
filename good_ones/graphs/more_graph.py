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

# Statistical calculations
pearson_corr, pearson_p = pearsonr(x, y)
spearman_corr, spearman_p = spearmanr(x, y)
slope, intercept, r_value, p_value_reg, std_err = stats.linregress(x, y)
predicted_linear = slope * x + intercept
residuals_linear = y - predicted_linear

# Piecewise regression analysis
def piecewise_linear(x, x0, a1, b1, a2, b2):
    """Piecewise linear function with breakpoint at x0"""
    return np.where(x < x0, a1*x + b1, a2*x + b2)

# Try different breakpoints around the critical zone
from scipy.optimize import curve_fit

best_r2 = 0
best_breakpoint = None
best_params = None
best_predicted = None

# Test breakpoints from 6.5 to 8.0
breakpoints_to_test = np.arange(6.5, 8.5, 0.1)

for breakpoint in breakpoints_to_test:
    try:
        # Fit piecewise model with fixed breakpoint
        def piecewise_fixed(x_val, a1, b1, a2, b2):
            return piecewise_linear(x_val, breakpoint, a1, b1, a2, b2)
        
        # Initial parameter guess
        popt, _ = curve_fit(piecewise_fixed, x, y, 
                           p0=[0.1, 0.5, -0.02, 0.3], 
                           maxfev=2000)
        
        # Calculate predictions and R²
        y_pred = piecewise_fixed(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        if r2 > best_r2:
            best_r2 = r2
            best_breakpoint = breakpoint
            best_params = popt
            best_predicted = y_pred
            
    except:
        continue

# Calculate piecewise residuals
residuals_piecewise = y - best_predicted if best_predicted is not None else residuals_linear

# Generate smooth curve for plotting
x_smooth_piece = np.linspace(x.min(), x.max(), 200)
if best_params is not None:
    y_smooth_piece = piecewise_linear(x_smooth_piece, best_breakpoint, *best_params)
else:
    y_smooth_piece = None

# Create the comprehensive figure
fig = plt.figure(figsize=(14, 10))

# Create a custom grid layout
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 0.3], hspace=0.4)

# Top plot - Main connection survival graph
ax1 = fig.add_subplot(gs[0, :])

# Main plot with enhanced styling
ax1.plot(x, y, 'o-', color='#2E86AB', linewidth=2.5, markersize=8, 
         markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=1.5,
         label='Connection Survival')

# Add trend line
z = np.polyfit(x, y, 2)  # 2nd degree polynomial fit
p = np.poly1d(z)
x_smooth = np.linspace(x.min(), x.max(), 100)
ax1.plot(x_smooth, p(x_smooth), '--', color='#F18F01', linewidth=2, alpha=0.8, label='Polynomial Trend')

# Add linear regression line
ax1.plot(x, predicted_linear, ':', color='green', linewidth=2, alpha=0.8, label='Linear Fit')

# Add piecewise regression line
if y_smooth_piece is not None:
    ax1.plot(x_smooth_piece, y_smooth_piece, '--', color='purple', linewidth=2, alpha=0.8, label='Piecewise Fit')
    ax1.axvline(x=best_breakpoint, color='purple', linestyle='--', alpha=0.6, 
               label=f'Breakpoint: {best_breakpoint:.1f}')

# Identify the steepest drop point
differences = np.diff(y)
steepest_drop_idx = np.argmin(differences)
ax1.axvline(x=x[steepest_drop_idx], color='red', linestyle=':', alpha=0.7, 
           label=f'Steepest Drop at Ratio {x[steepest_drop_idx]}')

# Add critical zones
ax1.axvspan(6, 7, alpha=0.2, color='green', label='Peak Performance Zone')
ax1.axvspan(7, 7.5, alpha=0.2, color='orange', label='Critical Transition Zone')
ax1.axvspan(7.5, 20, alpha=0.2, color='red', label='Post-Collapse Zone')

# Styling
ax1.set_xlabel('pBDNF/mBDNF Ratio', fontsize=12, fontweight='bold')
ax1.set_ylabel('Connection Survival', fontsize=12, fontweight='bold')
ax1.set_title('Connection Survival vs pBDNF/mBDNF Ratio Analysis', fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.legend(framealpha=0.9, fontsize=9, loc='upper right')

# Add text annotation for the critical point

# Middle - Residuals plot
ax2 = fig.add_subplot(gs[1, :])

# Plot both linear and piecewise residuals
ax2.scatter(x, residuals_linear, color='green', alpha=0.7, s=60, edgecolors='black', 
           linewidth=0.5, label='Linear Residuals')
if best_predicted is not None:
    ax2.scatter(x, residuals_piecewise, color='purple', alpha=0.7, s=60, edgecolors='black', 
               linewidth=0.5, label='Piecewise Residuals')

ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('pBDNF/mBDNF Ratio', fontweight='bold')
ax2.set_ylabel('Residuals', fontweight='bold')
ax2.set_title('Residuals Analysis: Linear vs Piecewise Models', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# Add residual statistics
residual_std_linear = np.std(residuals_linear)
residual_std_piecewise = np.std(residuals_piecewise) if best_predicted is not None else residual_std_linear

stats_box_text = f'Linear Residual Std: {residual_std_linear:.3f}\n'
if best_predicted is not None:
    stats_box_text += f'Piecewise Residual Std: {residual_std_piecewise:.3f}'

ax2.text(0.05, 0.95, stats_box_text, transform=ax2.transAxes, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
         fontsize=9, verticalalignment='top')

# Bottom - Statistical summary text
ax3 = fig.add_subplot(gs[2, :])
ax3.axis('off')

# Create comprehensive statistical summary
piecewise_info = ""
if best_predicted is not None:
    piecewise_info = f"Piecewise R² = {best_r2:.3f}, Breakpoint = {best_breakpoint:.1f}, "

stats_text = f"""STATISTICAL ANALYSIS SUMMARY:
Pearson Correlation: r = {pearson_corr:.3f}, p = {pearson_p:.4f} ({'Significant' if pearson_p < 0.05 else 'Not Significant'})
Spearman Correlation: ρ = {spearman_corr:.3f}, p = {spearman_p:.4f} ({'Significant' if spearman_p < 0.05 else 'Not Significant'})
Linear Regression: R² = {r_value**2:.3f}, p = {p_value_reg:.4f}, Slope = {slope:.4f} ± {std_err:.4f}
{piecewise_info}Improvement = {(best_r2 - r_value**2)*100:.1f}% better fit
Critical Findings: Peak at Ratio {x[np.argmax(y)]:.1f} ({np.max(y):.3f}), Steepest Drop at {x[steepest_drop_idx]:.2f}, Post-collapse plateau at {np.mean(y[y < 0.3]):.3f}"""

ax3.text(0.5, 0.5, stats_text, transform=ax3.transAxes, fontsize=11,
         ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
         fontweight='bold')

plt.suptitle('Comprehensive Connection Survival Analysis: pBDNF/mBDNF Ratio Effects', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.show()

# Print detailed statistical results
print("=== DETAILED STATISTICAL ANALYSIS ===")
print(f"\nDESCRIPTIVE STATISTICS:")
print(f"Sample Size: {len(x)} observations")
print(f"Ratio Range: {x.min():.1f} to {x.max():.1f}")
print(f"Mean Connection Survival: {np.mean(y):.4f} ± {np.std(y):.4f}")
print(f"Peak Performance: {np.max(y):.3f} at Ratio {x[np.argmax(y)]:.1f}")
print(f"Minimum Performance: {np.min(y):.3f} at Ratio {x[np.argmin(y)]:.1f}")

print(f"\nCORRELATION ANALYSIS:")
print(f"Pearson (Linear): r = {pearson_corr:.4f}, p = {pearson_p:.4f}")
print(f"Spearman (Monotonic): ρ = {spearman_corr:.4f}, p = {spearman_p:.4f}")
print(f"Interpretation: {'Strong' if abs(spearman_corr) > 0.5 else 'Moderate'} negative monotonic relationship")

print(f"\nREGRESSION ANALYSIS:")
print(f"Linear R-squared: {r_value**2:.4f} ({r_value**2*100:.1f}% variance explained)")
if best_predicted is not None:
    print(f"Piecewise R-squared: {best_r2:.4f} ({best_r2*100:.1f}% variance explained)")
    print(f"Model improvement: {(best_r2 - r_value**2)*100:.1f}% better fit")
    print(f"Optimal breakpoint: {best_breakpoint:.2f}")
    
    # Calculate slopes for each segment
    if best_params is not None:
        slope1, intercept1, slope2, intercept2 = best_params
        print(f"Pre-breakpoint slope: {slope1:.4f}")
        print(f"Post-breakpoint slope: {slope2:.4f}")
        
print(f"Linear slope: {slope:.6f} ± {std_err:.6f}")
print(f"Model adequacy: {'Poor' if r_value**2 < 0.3 else 'Good'} linear fit suggests piecewise relationship")

print(f"\nCRITICAL THRESHOLD ANALYSIS:")
steepest_change = abs(differences[steepest_drop_idx])
print(f"Steepest decline: {steepest_change:.3f} units between ratios {x[steepest_drop_idx]:.2f}-{x[steepest_drop_idx+1]:.2f}")
print(f"Represents {steepest_change/y[steepest_drop_idx]*100:.1f}% decrease from baseline")
print(f"Post-collapse stability: Mean survival = {np.mean(y[8:]):.3f} for ratios ≥ 8.0")

if best_predicted is not None:
    print(f"\nPIECEWISE MODEL INSIGHTS:")
    print(f"Optimal breakpoint at ratio {best_breakpoint:.2f} (close to observed critical zone)")
    print(f"Residual improvement: {((residual_std_linear - residual_std_piecewise)/residual_std_linear)*100:.1f}% reduction in error")
    print(f"Piecewise model captures the threshold effect much better than linear model")