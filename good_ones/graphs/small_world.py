import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

def generate_small_world_graph(ratios, survival_rates, small_world_coeffs, 
                              save_path="small_world_analysis.png", show_plot=True):
    """
    Generate a comprehensive small-world coefficient analysis graph
    
    Args:
        ratios: List/array of pBDNF/mBDNF ratios (e.g., [7.6, 7.8, 8.5, ...])
        survival_rates: List/array of connection survival percentages (e.g., [85.2, 78.3, 45.1, ...])
        small_world_coeffs: List/array of small-world coefficients (clustering/path_length)
        save_path: Path to save the figure
        show_plot: Whether to display the plot
    
    Returns:
        Dictionary with analysis results
    """
    
    # Convert to numpy arrays for easier manipulation
    ratios = np.array(ratios)
    survival_rates = np.array(survival_rates)
    small_world_coeffs = np.array(small_world_coeffs)
    
    # Validate input lengths
    if not (len(ratios) == len(survival_rates) == len(small_world_coeffs)):
        raise ValueError("All input arrays must have the same length")
    
    # Create the figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Main color scheme
    colors = plt.cm.viridis(np.linspace(0, 1, len(ratios)))
    
    # Plot 1: Small-World Coefficient vs Ratio (main plot)
    scatter = ax1.scatter(ratios, small_world_coeffs, c=survival_rates, s=120, 
                         cmap='RdYlBu', edgecolors='black', linewidth=1.5, alpha=0.8)
    
    # Add horizontal line at y=1 (random network threshold)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                label='Random Network (SW=1)')
    
    # Add trend line
    z = np.polyfit(ratios, small_world_coeffs, 2)  # 2nd degree polynomial
    p = np.poly1d(z)
    x_smooth = np.linspace(ratios.min(), ratios.max(), 100)
    ax1.plot(x_smooth, p(x_smooth), '--', color='darkblue', linewidth=2, 
             alpha=0.8, label='Polynomial Trend')
    
    # Identify critical points
    # Find where small-world coefficient drops most dramatically
    sw_changes = np.abs(np.diff(small_world_coeffs))
    if len(sw_changes) > 0:
        critical_idx = np.argmax(sw_changes)
        critical_ratio = ratios[critical_idx]
        critical_sw = small_world_coeffs[critical_idx]
        
    
    ax1.set_xlabel('pBDNF/mBDNF Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Small-World Coefficient', fontsize=12, fontweight='bold')

    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Connection Survival (%)', fontweight='bold')
    
    # Plot 2: Small-World vs Survival Rate (direct relationship)
    ax2.scatter(small_world_coeffs, survival_rates, c=ratios, s=120, 
               cmap='plasma', edgecolors='black', linewidth=1.5, alpha=0.8)
    
    # Add trend line for SW vs survival
    z2 = np.polyfit(small_world_coeffs, survival_rates, 1)  # Linear fit
    p2 = np.poly1d(z2)
    sw_smooth = np.linspace(small_world_coeffs.min(), small_world_coeffs.max(), 100)
    ax2.plot(sw_smooth, p2(sw_smooth), '--', color='red', linewidth=2, 
             alpha=0.8, label='Linear Trend')
    
    ax2.set_xlabel('Small-World Coefficient', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Connection Survival (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Network Survival vs Small-World Properties', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label('pBDNF/mBDNF Ratio', fontweight='bold')
    
    # Plot 3: Network Regime Classification
    ax3.bar(range(len(ratios)), small_world_coeffs, 
           color=['lightgreen' if sw > 1.5 else 'yellow' if sw > 1.0 else 'orange' if sw > 0.5 else 'red' 
                  for sw in small_world_coeffs],
           edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add regime threshold lines
    ax3.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, label='Strong Small-World (>1.5)')
    ax3.axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='Small-World (>1.0)')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Weak Small-World (>0.5)')
    
    ax3.set_xlabel('Data Point Index', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Small-World Coefficient', fontsize=12, fontweight='bold')
    ax3.set_title('Network Regime Classification', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(ratios)))
    ax3.set_xticklabels([f'{r:.1f}' for r in ratios], rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Statistical Summary and Analysis
    ax4.axis('off')  # Turn off axes for text plot
    
    # Calculate correlations and statistics
    sw_surv_corr, sw_surv_p = pearsonr(small_world_coeffs, survival_rates)
    sw_ratio_corr, sw_ratio_p = pearsonr(small_world_coeffs, ratios)
    
    # Network regime counts
    strong_sw_count = np.sum(small_world_coeffs > 1.5)
    sw_count = np.sum(small_world_coeffs > 1.0)
    weak_sw_count = np.sum(small_world_coeffs > 0.5)
    random_count = len(small_world_coeffs) - weak_sw_count
    
    # Find tipping point (where SW drops below 1.0)
    tipping_ratios = ratios[small_world_coeffs < 1.0]
    tipping_point = tipping_ratios[0] if len(tipping_ratios) > 0 else "Not reached"
    
    # Create statistical summary text
    stats_text = f"""SMALL-WORLD NETWORK ANALYSIS
    
📊 Basic Statistics:
   • SW Coefficient Range: {small_world_coeffs.min():.3f} - {small_world_coeffs.max():.3f}
   • Mean SW Coefficient: {small_world_coeffs.mean():.3f} ± {small_world_coeffs.std():.3f}
   • Median SW Coefficient: {np.median(small_world_coeffs):.3f}

🔗 Correlations:
   • SW vs Survival: r = {sw_surv_corr:.3f}, p = {sw_surv_p:.4f}
   • SW vs Ratio: r = {sw_ratio_corr:.3f}, p = {sw_ratio_p:.4f}

🏗️ Network Regimes:
   • Strong Small-World (>1.5): {strong_sw_count}/{len(ratios)} networks
   • Small-World (>1.0): {sw_count}/{len(ratios)} networks  
   • Weak Small-World (>0.5): {weak_sw_count}/{len(ratios)} networks
   • Random-like (≤0.5): {random_count}/{len(ratios)} networks

🎯 Critical Points:
   • SW Tipping Point: Ratio {tipping_point}
   • Most Vulnerable SW: {small_world_coeffs.min():.3f} at Ratio {ratios[np.argmin(small_world_coeffs)]:.1f}
   • Most Robust SW: {small_world_coeffs.max():.3f} at Ratio {ratios[np.argmax(small_world_coeffs)]:.1f}

🧠 Interpretation:
   • {'Strong' if sw_surv_corr > 0.5 else 'Moderate' if sw_surv_corr > 0.3 else 'Weak'} correlation with survival
   • Network maintains small-world properties until ratio ~{tipping_point}
   • {'High' if np.mean(small_world_coeffs) > 1.0 else 'Low'} overall small-world character
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Overall figure title
    plt.suptitle('Comprehensive Small-World Network Analysis: BDNF Ratio Effects', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Small-world analysis saved to: {save_path}")
    
    # Show the plot
    if show_plot:
        sns.despine()
        plt.show()
    else:
        plt.close()
    
    # Return analysis results
    results = {
        'sw_survival_correlation': sw_surv_corr,
        'sw_survival_p_value': sw_surv_p,
        'sw_ratio_correlation': sw_ratio_corr,
        'sw_ratio_p_value': sw_ratio_p,
        'mean_small_world': small_world_coeffs.mean(),
        'std_small_world': small_world_coeffs.std(),
        'tipping_point': tipping_point,
        'strong_sw_count': strong_sw_count,
        'sw_count': sw_count,
        'weak_sw_count': weak_sw_count,
        'random_count': random_count,
        'most_robust_ratio': ratios[np.argmax(small_world_coeffs)],
        'most_vulnerable_ratio': ratios[np.argmin(small_world_coeffs)],
        'critical_transition_ratio': critical_ratio if 'critical_ratio' in locals() else None
    }
    
    return results

def quick_small_world_plot(ratios, survival_rates, small_world_coeffs, title="Small-World Analysis"):
    """
    Generate a quick single-panel small-world plot similar to your original graphs
    
    Args:
        ratios: pBDNF/mBDNF ratios
        survival_rates: Connection survival percentages  
        small_world_coeffs: Small-world coefficients
        title: Plot title
    """
    
    plt.figure(figsize=(12, 8))
    
    # Main scatter plot
    scatter = plt.scatter(ratios, small_world_coeffs, c=survival_rates, s=120, 
                         cmap='RdYlBu', edgecolors='black', linewidth=1.5, alpha=0.8)
    
    # Add trend line
    z = np.polyfit(ratios, small_world_coeffs, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(ratios), max(ratios), 100)
    plt.plot(x_smooth, p(x_smooth), '--', color='darkblue', linewidth=2, alpha=0.8)
    
    # Add reference line at SW = 1
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                label='Random Network Threshold')
    
    # Formatting
    plt.xlabel('pBDNF/mBDNF Ratio', fontsize=12, fontweight='bold')
    plt.ylabel('Small-World Coefficient', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Connection Survival (%)', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Quick stats
    sw_surv_corr, sw_surv_p = pearsonr(small_world_coeffs, survival_rates)
    print(f"\n📊 Quick Analysis:")
    print(f"   Small-World vs Survival Correlation: r = {sw_surv_corr:.3f}, p = {sw_surv_p:.4f}")
    print(f"   Mean Small-World Coefficient: {np.mean(small_world_coeffs):.3f}")
    print(f"   Networks with SW > 1.0: {np.sum(np.array(small_world_coeffs) > 1.0)}/{len(small_world_coeffs)}")

# Example usage and test data
if __name__ == "__main__":
    print("🔬 Small-World Graph Generator - Example Usage")
    
    # Example data (replace with your real data)
    example_survival = [0.076, 0.201, 0.187, 0.512, 0.443, 0.746, 0.942, 0.989, 0.994, 0.870, 0.795, 0.775, 0.765, 0.759, 0.754, 0.785, 0.762, 0.756, 0.726, 0.782, 0.763, 0.620, 0.546, 0.465, 0.331, 0.394, 0.231, 0.215, 0.245, 0.256, 0.220, 0.241, 0.251, 0.225, 0.281, 0.239, 0.261, 0.250,  0.233, 0.218]
    example_ratios = [3.000, 3.200, 3.400, 3.600, 3.800, 4.000, 4.200, 4.400, 4.600, 4.800, 5.000, 5.200, 5.400, 5.600, 5.800, 6.000, 6.200, 6.400, 6.600, 6.800, 7.000, 7.200, 7.300, 7.400, 7.500, 7.600, 7.800, 8.000, 8.500, 9.000, 9.500, 10.00, 10.50, 11.00, 11.50, 12.00, 12.50, 13.00,  15.00, 20.00]
    example_small_world = [0.0358, 0.0734, 0.0749, 0.1154, 0.1071, 0.1271, 0.1510, 0.1374, 0.1563, 0.1764, 0.1188, 0.1342, 0.1376, 0.1560, 0.1401, 0.1050, 0.1364, 0.1320, 0.1637, 0.1514, 0.1111, 0.1196, 0.0942, 0.0874, 0.0744, 0.0618, 0.0473, 0.0485, 0.0286, 0.0634, 0.0424, 0.0240, 0.0491, 0.0400, 0.0382, 0.0759, 0.0744, 0.0473, 0.0475, 0.0442]
    
    print("\n📊 Generating comprehensive small-world analysis...")
    results = generate_small_world_graph(
        ratios=example_ratios,
        survival_rates=example_survival, 
        small_world_coeffs=example_small_world,
        save_path="example_small_world_analysis.png",
        show_plot=True
    )
    
    print(f"\n✅ Analysis complete!")
    print(f"📈 Tipping point detected at ratio: {results['tipping_point']}")
    print(f"🔗 Small-world vs survival correlation: {results['sw_survival_correlation']:.3f}")
    
    print(f"\n🎯 To use with your data, simply call:")
    print(f"   results = generate_small_world_graph(your_ratios, your_survival, your_sw_coeffs)")
    
    print(f"\n🚀 For a quick single plot, use:")  
    print(f"   quick_small_world_plot(your_ratios, your_survival, your_sw_coeffs)")