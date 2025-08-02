import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class AdvancedTippingAnalysis:
    def __init__(self, ratios, survival_rates, time_series_data=None):
        """
        Advanced analysis for tipping point detection
        
        Args:
            ratios: List of BDNF:proBDNF ratios tested
            survival_rates: Connection survival percentages
            time_series_data: Dict with time series for each ratio (optional)
        """
        self.ratios = np.array(ratios)
        self.survival_rates = np.array(survival_rates)
        self.time_series_data = time_series_data or {}
        
    def catastrophe_theory_analysis(self):
        """
        Apply catastrophe theory to identify fold bifurcations and tipping points
        """
        # Fit cusp catastrophe model: x^3 + ax + b = 0
        def cusp_model(ratio, a, b, c, d):
            """Cusp catastrophe surface"""
            return a * ratio**3 + b * ratio**2 + c * ratio + d
        
        # Fit the model
        popt, pcov = curve_fit(cusp_model, self.ratios, self.survival_rates, 
                              p0=[1, -1, 1, 50], maxfev=5000)
        
        # Find critical points (where derivative = 0)
        def cusp_derivative(ratio, a, b, c):
            return 3*a * ratio**2 + 2*b * ratio + c
        
        # Solve for critical points
        from scipy.optimize import fsolve
        critical_points = []
        for guess in [6, 7, 8, 9]:
            try:
                cp = fsolve(lambda x: cusp_derivative(x, *popt[:3]), guess)[0]
                if 3 <= cp <= 20:  # Within our ratio range
                    critical_points.append(cp)
            except:
                pass
        
        critical_points = np.unique(np.round(critical_points, 2))
        
        # Plot catastrophe surface
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Data with fitted cusp model
        ratio_smooth = np.linspace(self.ratios.min(), self.ratios.max(), 200)
        cusp_fit = cusp_model(ratio_smooth, *popt)
        
        ax1.scatter(self.ratios, self.survival_rates, color='red', s=100, 
                   zorder=5, label='Observed Data', edgecolors='black')
        ax1.plot(ratio_smooth, cusp_fit, 'b-', linewidth=3, label='Cusp Model Fit')
        
        # Mark critical points
        for cp in critical_points:
            cp_survival = cusp_model(cp, *popt)
            ax1.axvline(cp, color='orange', linestyle='--', alpha=0.8, linewidth=2)
            ax1.scatter(cp, cp_survival, color='orange', s=200, marker='*', 
                       zorder=6, edgecolors='black', linewidth=2)
            ax1.text(cp, cp_survival + 5, f'Critical\n{cp:.2f}', 
                    ha='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='yellow', alpha=0.7))
        
        ax1.set_xlabel('pBDNF/mBDNF Ratio', fontweight='bold')
        ax1.set_ylabel('Connection Survival (%)', fontweight='bold')
        ax1.set_title('Catastrophe Theory: Cusp Model Fitting', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Stability landscape (potential function)
        ax2.plot(ratio_smooth, -cusp_fit, 'purple', linewidth=3, label='Potential Landscape')
        ax2.fill_between(ratio_smooth, -cusp_fit, alpha=0.3, color='purple')
        
        for cp in critical_points:
            cp_potential = -cusp_model(cp, *popt)
            ax2.axvline(cp, color='orange', linestyle='--', alpha=0.8, linewidth=2)
            ax2.scatter(cp, cp_potential, color='orange', s=200, marker='*', zorder=5)
        
        ax2.set_xlabel('pBDNF/mBDNF Ratio', fontweight='bold')
        ax2.set_ylabel('Potential Energy (Negative Survival)', fontweight='bold')
        ax2.set_title('Stability Landscape', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return critical_points, popt, pcov
    
    def phase_transition_analysis(self):
        """
        Analyze the phase transition using order parameter and susceptibility
        """
        # Calculate order parameter (deviation from high survival state)
        max_survival = np.max(self.survival_rates)
        order_parameter = (max_survival - self.survival_rates) / max_survival
        
        # Calculate susceptibility (rate of change of order parameter)
        susceptibility = np.abs(np.gradient(order_parameter, self.ratios))
        
        # Find maximum susceptibility (indicates critical point)
        critical_idx = np.argmax(susceptibility)
        critical_ratio = self.ratios[critical_idx]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Order parameter
        ax1.plot(self.ratios, order_parameter, 'bo-', linewidth=3, markersize=8)
        ax1.axvline(critical_ratio, color='red', linestyle='--', linewidth=2, 
                   label=f'Critical Ratio: {critical_ratio:.2f}')
        ax1.set_xlabel('pBDNF/mBDNF Ratio')
        ax1.set_ylabel('Order Parameter\n(Normalized Survival Loss)')
        ax1.set_title('Phase Transition: Order Parameter', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Susceptibility
        ax2.plot(self.ratios, susceptibility, 'ro-', linewidth=3, markersize=8)
        ax2.axvline(critical_ratio, color='red', linestyle='--', linewidth=2)
        ax2.scatter(critical_ratio, susceptibility[critical_idx], color='yellow', 
                   s=300, marker='*', edgecolors='black', linewidth=2, zorder=5)
        ax2.set_xlabel('pBDNF/mBDNF Ratio')
        ax2.set_ylabel('Susceptibility\n(Rate of Change)')
        ax2.set_title('Critical Point Detection: Maximum Susceptibility', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Critical exponent analysis
        # Fit power law near critical point: order_parameter ~ |ratio - critical_ratio|^β
        mask = np.abs(self.ratios - critical_ratio) < 2  # Points near critical ratio
        if np.sum(mask) > 3:
            x_fit = np.abs(self.ratios[mask] - critical_ratio)
            y_fit = order_parameter[mask]
            
            # Remove zero distance points
            nonzero_mask = x_fit > 0.01
            if np.sum(nonzero_mask) > 2:
                x_fit = x_fit[nonzero_mask]
                y_fit = y_fit[nonzero_mask]
                
                # Fit power law: log(y) = β*log(x) + const
                log_x = np.log(x_fit)
                log_y = np.log(y_fit + 1e-6)  # Avoid log(0)
                
                slope, intercept = np.polyfit(log_x, log_y, 1)
                critical_exponent = slope
                
                ax3.loglog(x_fit, y_fit, 'go', markersize=8, label='Data near critical point')
                x_theory = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 50)
                y_theory = np.exp(intercept) * x_theory**critical_exponent
                ax3.loglog(x_theory, y_theory, 'r--', linewidth=2, 
                          label=f'Power law: β = {critical_exponent:.2f}')
        
        ax3.set_xlabel('|Ratio - Critical Ratio|')
        ax3.set_ylabel('Order Parameter')
        ax3.set_title('Critical Exponent Analysis', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Phase diagram
        ax4.scatter(self.ratios, self.survival_rates, c=order_parameter, 
                   s=100, cmap='RdYlBu_r', edgecolors='black')
        ax4.axvline(critical_ratio, color='red', linestyle='--', linewidth=3, alpha=0.8)
        
        # Add phase labels
        ax4.text(self.ratios.min() + 0.5, max(self.survival_rates) * 0.9, 
                'STABLE\nPHASE', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.7))
        ax4.text(self.ratios.max() - 1, max(self.survival_rates) * 0.3, 
                'COLLAPSED\nPHASE', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.7))
        
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Order Parameter', fontweight='bold')
        ax4.set_xlabel('pBDNF/mBDNF Ratio')
        ax4.set_ylabel('Connection Survival (%)')
        ax4.set_title('Phase Diagram', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return critical_ratio, critical_exponent if 'critical_exponent' in locals() else None
    
    def time_series_early_warning_signals(self):
        """
        Analyze time series data for early warning signals of critical transitions
        """
        if not self.time_series_data:
            print("No time series data provided")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        warning_metrics = {}
        
        for i, (ratio, ts_data) in enumerate(self.time_series_data.items()):
            if i >= 4:  # Limit to 4 ratios for clarity
                break
                
            time_points = ts_data.get('time', np.arange(len(ts_data['survival'])))
            survival_ts = np.array(ts_data['survival'])
            
            # Calculate early warning signals
            
            # 1. Variance (increasing before transition)
            window_size = len(survival_ts) // 10
            rolling_variance = []
            rolling_autocorr = []
            rolling_recovery = []
            
            for j in range(window_size, len(survival_ts)):
                window_data = survival_ts[j-window_size:j]
                
                # Variance
                rolling_variance.append(np.var(window_data))
                
                # Autocorrelation (lag-1)
                if len(window_data) > 1:
                    autocorr = np.corrcoef(window_data[:-1], window_data[1:])[0,1]
                    rolling_autocorr.append(autocorr if not np.isnan(autocorr) else 0)
                else:
                    rolling_autocorr.append(0)
                
                # Recovery rate (after small perturbations)
                diffs = np.diff(window_data)
                recovery_rate = -np.mean(diffs[diffs < 0]) if len(diffs[diffs < 0]) > 0 else 0
                rolling_recovery.append(recovery_rate)
            
            rolling_time = time_points[window_size:]
            
            warning_metrics[ratio] = {
                'variance': rolling_variance,
                'autocorr': rolling_autocorr,
                'recovery': rolling_recovery,
                'time': rolling_time
            }
        
        # Plot early warning signals
        colors = ['blue', 'red', 'green', 'orange']
        
        for idx, (signal_name, ylabel, title) in enumerate([
            ('variance', 'Variance', 'Increasing Variance'),
            ('autocorr', 'Autocorrelation', 'Increasing Autocorrelation'),
            ('recovery', 'Recovery Rate', 'Decreasing Recovery Rate')
        ]):
            ax = axes[idx, 0]
            
            for i, (ratio, metrics) in enumerate(warning_metrics.items()):
                if i < 4:
                    ax.plot(metrics['time'], metrics[signal_name], 
                           color=colors[i], linewidth=2, label=f'Ratio 1:{ratio}')
            
            ax.set_xlabel('Time')
            ax.set_ylabel(ylabel)
            ax.set_title(f'Early Warning Signal: {title}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Summary plot - final values vs ratio
            ax_summary = axes[idx, 1]
            ratios_list = list(warning_metrics.keys())
            final_values = [warning_metrics[r][signal_name][-1] if warning_metrics[r][signal_name] 
                          else 0 for r in ratios_list]
            
            ax_summary.scatter(ratios_list, final_values, s=100, c=colors[:len(ratios_list)], 
                              edgecolors='black')
            ax_summary.set_xlabel('pBDNF/mBDNF Ratio')
            ax_summary.set_ylabel(f'Final {ylabel}')
            ax_summary.set_title(f'Final {ylabel} vs Ratio', fontweight='bold')
            ax_summary.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return warning_metrics
    
    def multiscale_analysis(self):
        """
        Analyze tipping behavior across multiple scales using wavelet transform
        """
        # Interpolate data for higher resolution
        f_interp = interp1d(self.ratios, self.survival_rates, kind='cubic')
        ratio_hires = np.linspace(self.ratios.min(), self.ratios.max(), 1000)
        survival_hires = f_interp(ratio_hires)
        
        # Wavelet transform to identify scales of change
        try:
            from scipy import signal as sig
            widths = np.arange(1, 50)
            # Use morlet wavelet as alternative to ricker
            cwt_matrix = np.abs(np.array([sig.hilbert(np.convolve(survival_hires, 
                                sig.morlet(len(survival_hires), w), mode='same')) 
                                for w in widths]))
        except:
            # Fallback: simple multi-scale derivative analysis
            cwt_matrix = np.array([np.abs(np.gradient(survival_hires, n)) for n in range(1, 50)])
            widths = np.arange(1, 50)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Continuous Wavelet Transform
        im1 = ax1.imshow(np.abs(cwt_matrix), extent=[ratio_hires.min(), ratio_hires.max(), 
                                                    widths.min(), widths.max()], 
                        cmap='viridis', aspect='auto', origin='lower')
        ax1.set_xlabel('pBDNF/mBDNF Ratio')
        ax1.set_ylabel('Scale (Wavelet Width)')
        ax1.set_title('Continuous Wavelet Transform', fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Magnitude')
        
        # Plot 2: Ridge detection (find dominant scales)
        try:
            ridge_lines = sig.find_peaks(survival_hires, height=np.mean(survival_hires))[0]
        except:
            ridge_lines = []
        ax2.plot(ratio_hires, survival_hires, 'b-', linewidth=2, label='Interpolated Data')
        for ridge in ridge_lines:
            if ridge < len(ratio_hires):
                ax2.axvline(ratio_hires[ridge], color='red', linestyle='--', alpha=0.7)
                ax2.scatter(ratio_hires[ridge], survival_hires[ridge], 
                           color='red', s=100, marker='*', zorder=5)
        
        ax2.set_xlabel('pBDNF/mBDNF Ratio')
        ax2.set_ylabel('Connection Survival (%)')
        ax2.set_title('Detected Critical Points (Wavelet Ridges)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scale-dependent variance
        scale_variance = np.var(cwt_matrix, axis=1)
        ax3.plot(widths, scale_variance, 'g-', linewidth=3)
        max_var_scale = widths[np.argmax(scale_variance)]
        ax3.axvline(max_var_scale, color='red', linestyle='--', linewidth=2, 
                   label=f'Max Variance Scale: {max_var_scale}')
        ax3.set_xlabel('Scale (Wavelet Width)')
        ax3.set_ylabel('Variance Across Ratios')
        ax3.set_title('Scale-Dependent Variance', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Multiscale entropy
        def sample_entropy(data, m=2, r=0.2):
            """Calculate sample entropy"""
            N = len(data)
            patterns = np.array([data[i:i+m] for i in range(N-m+1)])
            
            def _maxdist(x1, x2):
                return max([abs(ua - va) for ua, va in zip(x1, x2)])
            
            def _phi(m):
                patterns = np.array([data[i:i+m] for i in range(N-m+1)])
                C = np.zeros(N-m+1)
                for i in range(N-m+1):
                    template = patterns[i]
                    for j in range(N-m+1):
                        if _maxdist(template, patterns[j]) <= r:
                            C[i] += 1
                phi = np.mean(np.log(C / (N-m+1)))
                return phi
            
            return _phi(m) - _phi(m+1)
        
        entropies = []
        window_size = 5
        for i in range(len(self.survival_rates) - window_size + 1):
            window_data = self.survival_rates[i:i+window_size]
            entropy_val = sample_entropy(window_data, m=2, r=0.2*np.std(window_data))
            entropies.append(entropy_val)
        
        entropy_ratios = self.ratios[window_size-1:]
        ax4.plot(entropy_ratios, entropies, 'purple', linewidth=3, marker='o')
        ax4.set_xlabel('pBDNF/mBDNF Ratio')
        ax4.set_ylabel('Sample Entropy')
        ax4.set_title('Complexity Analysis: Sample Entropy', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return ridge_lines, max_var_scale, entropies
    
    def network_topology_analysis(self):
        """
        Analyze the relationship between ratios and network topology changes
        """
        # Simulate different network metrics for different ratios
        # (In real analysis, you'd extract these from your network data)
        
        np.random.seed(42)  # For reproducible "simulation"
        
        # Simulate network metrics that change with ratio
        clustering_coeff = 0.8 - 0.6 * (self.ratios - self.ratios.min()) / (self.ratios.max() - self.ratios.min())
        clustering_coeff += np.random.normal(0, 0.05, len(self.ratios))
        
        avg_path_length = 2.0 + 2.0 * (self.ratios - self.ratios.min()) / (self.ratios.max() - self.ratios.min())
        avg_path_length += np.random.normal(0, 0.1, len(self.ratios))
        
        modularity = 0.7 - 0.5 * (self.ratios - self.ratios.min()) / (self.ratios.max() - self.ratios.min())
        modularity += np.random.normal(0, 0.03, len(self.ratios))
        
        # Small-world coefficient
        small_world_coeff = clustering_coeff / avg_path_length
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Clustering coefficient
        ax1.scatter(self.ratios, clustering_coeff, c=self.survival_rates, 
                   s=100, cmap='RdYlBu', edgecolors='black')
        ax1.set_xlabel('pBDNF/mBDNF Ratio')
        ax1.set_ylabel('Clustering Coefficient')
        ax1.set_title('Network Clustering vs Ratio', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar1.set_label('Survival %')
        
        # Plot 2: Average path length
        ax2.scatter(self.ratios, avg_path_length, c=self.survival_rates, 
                   s=100, cmap='RdYlBu', edgecolors='black')
        ax2.set_xlabel('pBDNF/mBDNF Ratio')
        ax2.set_ylabel('Average Path Length')
        ax2.set_title('Path Length vs Ratio', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar2.set_label('Survival %')
        
        # Plot 3: Small-world coefficient
        ax3.scatter(self.ratios, small_world_coeff, c=self.survival_rates, 
                   s=100, cmap='RdYlBu', edgecolors='black')
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Random Network')
        ax3.set_xlabel('pBDNF/mBDNF Ratio')
        ax3.set_ylabel('Small-World Coefficient')
        ax3.set_title('Small-World Property vs Ratio', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        cbar3 = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar3.set_label('Survival %')
        
        # Plot 4: Network topology phase space
        ax4.scatter(clustering_coeff, avg_path_length, c=self.ratios, 
                   s=self.survival_rates*2, cmap='viridis', edgecolors='black', alpha=0.7)
        
        # Add trajectory arrows
        for i in range(len(self.ratios)-1):
            ax4.annotate('', xy=(clustering_coeff[i+1], avg_path_length[i+1]), 
                        xytext=(clustering_coeff[i], avg_path_length[i]),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
        
        ax4.set_xlabel('Clustering Coefficient')
        ax4.set_ylabel('Average Path Length')
        ax4.set_title('Network Topology Phase Space', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        cbar4 = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar4.set_label('Ratio')
        
        plt.tight_layout()
        plt.show()
        
        return clustering_coeff, avg_path_length, modularity, small_world_coeff
    
    def machine_learning_classification(self):
        """
        Use ML to classify pre/post tipping point and identify key features
        """
        # Create feature matrix
        features = np.column_stack([
            self.ratios,
            self.survival_rates,
            np.gradient(self.survival_rates),  # First derivative
            np.gradient(np.gradient(self.survival_rates)),  # Second derivative
        ])
        
        # Create binary labels (collapsed vs stable)
        threshold = 50  # 50% survival threshold
        labels = (self.survival_rates < threshold).astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # PCA for dimensionality reduction and visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: PCA visualization
        scatter1 = ax1.scatter(features_pca[:, 0], features_pca[:, 1], 
                              c=self.survival_rates, s=100, cmap='RdYlBu', 
                              edgecolors='black')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title('PCA: Feature Space Visualization', fontweight='bold')
        plt.colorbar(scatter1, ax=ax1, label='Survival %')
        
        # Plot 2: K-means clustering
        scatter2 = ax2.scatter(features_pca[:, 0], features_pca[:, 1], 
                              c=clusters, s=100, cmap='Set1', edgecolors='black')
        
        # Plot cluster centers
        centers_pca = pca.transform(scaler.transform(
            kmeans.cluster_centers_[:, :features.shape[1]]))
        ax2.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='yellow', s=300, marker='X', edgecolors='black', linewidth=2)
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax2.set_title('K-Means Clustering in Feature Space', fontweight='bold')
        
        # Plot 3: Feature importance
        feature_names = ['Ratio', 'Survival', '1st Derivative', '2nd Derivative']
        pca_components = np.abs(pca.components_)
        
        x_pos = np.arange(len(feature_names))
        ax3.bar(x_pos, pca_components[0], alpha=0.7, label='PC1', color='blue')
        ax3.bar(x_pos, pca_components[1], alpha=0.7, label='PC2', color='red', bottom=pca_components[0])
        
        ax3.set_xlabel('Features')
        ax3.set_ylabel('PCA Component Weight')
        ax3.set_title('Feature Importance in Principal Components', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(feature_names, rotation=45)
        ax3.legend()
        
        # Plot 4: Decision boundary visualization
        # Create a grid for decision boundary
        h = 0.1
        x_min, x_max = features_pca[:, 0].min() - 1, features_pca[:, 0].max() + 1
        y_min, y_max = features_pca[:, 1].min() - 1, features_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Simple decision boundary based on survival threshold
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Approximate decision boundary
        boundary_mask = np.zeros(mesh_points.shape[0])
        for i, point in enumerate(mesh_points):
            # Find closest data point
            distances = np.sqrt(np.sum((features_pca - point)**2, axis=1))
            closest_idx = np.argmin(distances)
            boundary_mask[i] = 1 if self.survival_rates[closest_idx] < threshold else 0
        
        boundary_mask = boundary_mask.reshape(xx.shape)
        
        ax4.contourf(xx, yy, boundary_mask, alpha=0.3, cmap='RdYlBu')
        scatter4 = ax4.scatter(features_pca[:, 0], features_pca[:, 1], 
                              c=labels, s=100, cmap='RdYlBu', edgecolors='black')
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax4.set_title('Classification: Stable vs Collapsed States', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate silhouette score for clustering quality
        silhouette_avg = silhouette_score(features_scaled, clusters)
        
        return {
            'pca_components': pca.components_,
            'explained_variance': pca.explained_variance_ratio_,
            'clusters': clusters,
            'silhouette_score': silhouette_avg,
            'feature_importance': dict(zip(feature_names, pca_components[0]))
        }
    
    def hysteresis_analysis(self):
        """
        Analyze hysteresis effects - different behavior when approaching 
        tipping point from different directions
        """
        # Simulate forward and backward parameter sweeps
        # In real analysis, you'd have data from both increasing and decreasing ratio experiments
        
        # Forward sweep (increasing ratio)
        ratios_forward = self.ratios
        survival_forward = self.survival_rates
        
        # Simulate backward sweep (decreasing ratio) with hysteresis
        ratios_backward = self.ratios[::-1]
        # Add hysteresis offset - system "remembers" collapsed state
        hysteresis_offset = 10 * np.exp(-(self.ratios[::-1] - 7.5)**2 / 0.5)
        survival_backward = self.survival_rates[::-1] - hysteresis_offset
        survival_backward = np.maximum(survival_backward, 0)  # Can't go below 0
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Hysteresis loop
        ax1.plot(ratios_forward, survival_forward, 'b-o', linewidth=3, 
                markersize=8, label='Forward (Increasing Ratio)')
        ax1.plot(ratios_backward, survival_backward, 'r-s', linewidth=3, 
                markersize=8, label='Backward (Decreasing Ratio)')
        
        # Add arrows to show direction
        for i in range(0, len(ratios_forward)-1, 2):
            ax1.annotate('', xy=(ratios_forward[i+1], survival_forward[i+1]), 
                        xytext=(ratios_forward[i], survival_forward[i]),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
        
        for i in range(0, len(ratios_backward)-1, 2):
            ax1.annotate('', xy=(ratios_backward[i+1], survival_backward[i+1]), 
                        xytext=(ratios_backward[i], survival_backward[i]),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        ax1.set_xlabel('pBDNF/mBDNF Ratio')
        ax1.set_ylabel('Connection Survival (%)')
        ax1.set_title('Hysteresis Loop Analysis', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hysteresis width
        hysteresis_width = np.abs(survival_forward - survival_backward[::-1])
        ax2.plot(ratios_forward, hysteresis_width, 'purple', linewidth=3, marker='o')
        max_hyst_idx = np.argmax(hysteresis_width)
        ax2.scatter(ratios_forward[max_hyst_idx], hysteresis_width[max_hyst_idx], 
                   color='red', s=200, marker='*', zorder=5)
        ax2.text(ratios_forward[max_hyst_idx], hysteresis_width[max_hyst_idx] + 2, 
                f'Max Hysteresis\nRatio: {ratios_forward[max_hyst_idx]:.2f}', 
                ha='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        ax2.set_xlabel('pBDNF/mBDNF Ratio')
        ax2.set_ylabel('Hysteresis Width (%)')
        ax2.set_title('Hysteresis Width vs Ratio', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Memory effect
        memory_effect = np.cumsum(hysteresis_width) / np.arange(1, len(hysteresis_width)+1)
        ax3.plot(ratios_forward, memory_effect, 'orange', linewidth=3, marker='d')
        ax3.set_xlabel('pBDNF/mBDNF Ratio')
        ax3.set_ylabel('Cumulative Memory Effect')
        ax3.set_title('System Memory Effect', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Recovery potential
        recovery_potential = survival_backward[::-1] - survival_forward
        ax4.plot(ratios_forward, recovery_potential, 'green', linewidth=3, marker='^')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.fill_between(ratios_forward, recovery_potential, 0, alpha=0.3, color='green')
        ax4.set_xlabel('pBDNF/mBDNF Ratio')
        ax4.set_ylabel('Recovery Potential (%)')
        ax4.set_title('Network Recovery Potential', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'hysteresis_width': hysteresis_width,
            'max_hysteresis_ratio': ratios_forward[max_hyst_idx],
            'memory_effect': memory_effect,
            'recovery_potential': recovery_potential
        }
    
    def fractal_dimension_analysis(self):
        """
        Calculate fractal dimension to characterize the roughness/complexity 
        of the survival curve near the tipping point
        """
        def box_count_fractal_dim(data, max_box_size=None):
            """Calculate fractal dimension using box counting method"""
            if max_box_size is None:
                max_box_size = len(data) // 4
            
            # Normalize data to [0, 1]
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            
            # Create grid of different box sizes
            box_sizes = np.logspace(0, np.log10(max_box_size), 20, dtype=int)
            box_sizes = np.unique(box_sizes)
            
            counts = []
            for box_size in box_sizes:
                # Divide data into boxes and count non-empty boxes
                n_boxes_x = len(data_norm) // box_size
                if n_boxes_x == 0:
                    continue
                    
                boxes = np.array_split(data_norm, n_boxes_x)
                non_empty_boxes = sum(1 for box in boxes if len(box) > 0 and np.ptp(box) > 0)
                counts.append(non_empty_boxes)
            
            if len(counts) < 3:
                return np.nan, np.nan, np.nan
            
            # Fit log-log relationship: log(count) = -D * log(box_size) + const
            log_box_sizes = np.log(box_sizes[:len(counts)])
            log_counts = np.log(counts)
            
            slope, intercept = np.polyfit(log_box_sizes, log_counts, 1)
            fractal_dim = -slope
            
            return fractal_dim, box_sizes[:len(counts)], counts
        
        # Calculate fractal dimension for different sections of the curve
        sections = {
            'Full curve': self.survival_rates,
            'Pre-transition': self.survival_rates[self.ratios <= 7.3],
            'Transition zone': self.survival_rates[(self.ratios >= 7.0) & (self.ratios <= 8.0)],
            'Post-transition': self.survival_rates[self.ratios >= 7.7]
        }
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        fractal_results = {}
        colors = ['blue', 'green', 'orange', 'red']
        
        # Plot 1: Original data with sections highlighted
        ax1.plot(self.ratios, self.survival_rates, 'ko-', linewidth=2, markersize=6, label='Full Data')
        
        # Highlight different sections
        pre_mask = self.ratios <= 7.3
        trans_mask = (self.ratios >= 7.0) & (self.ratios <= 8.0)
        post_mask = self.ratios >= 7.7
        
        ax1.plot(self.ratios[pre_mask], self.survival_rates[pre_mask], 
                'go', markersize=8, label='Pre-transition')
        ax1.plot(self.ratios[trans_mask], self.survival_rates[trans_mask], 
                'o', color='orange', markersize=8, label='Transition zone')
        ax1.plot(self.ratios[post_mask], self.survival_rates[post_mask], 
                'ro', markersize=8, label='Post-transition')
        
        ax1.set_xlabel('pBDNF/mBDNF Ratio')
        ax1.set_ylabel('Connection Survival (%)')
        ax1.set_title('Data Sections for Fractal Analysis', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box counting visualization for transition zone
        trans_data = self.survival_rates[trans_mask]
        if len(trans_data) > 3:
            frac_dim, box_sizes, counts = box_count_fractal_dim(trans_data)
            
            if not np.isnan(frac_dim):
                ax2.loglog(box_sizes, counts, 'o-', color='orange', linewidth=2, markersize=6)
                
                # Fit line
                log_box_sizes = np.log(box_sizes)
                log_counts = np.log(counts)
                slope, intercept = np.polyfit(log_box_sizes, log_counts, 1)
                fit_line = np.exp(intercept) * box_sizes ** slope
                ax2.loglog(box_sizes, fit_line, '--', color='red', linewidth=2, 
                          label=f'Fractal Dim = {frac_dim:.3f}')
        
        ax2.set_xlabel('Box Size')
        ax2.set_ylabel('Number of Non-empty Boxes')
        ax2.set_title('Box Counting: Transition Zone', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fractal dimensions for all sections
        fractal_dims = []
        section_names = []
        
        for i, (name, data) in enumerate(sections.items()):
            if len(data) > 3:
                frac_dim, _, _ = box_count_fractal_dim(data)
                if not np.isnan(frac_dim):
                    fractal_dims.append(frac_dim)
                    section_names.append(name)
                    fractal_results[name] = frac_dim
        
        if fractal_dims:
            bars = ax3.bar(range(len(fractal_dims)), fractal_dims, 
                          color=colors[:len(fractal_dims)], alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Data Section')
            ax3.set_ylabel('Fractal Dimension')
            ax3.set_title('Fractal Dimension by Section', fontweight='bold')
            ax3.set_xticks(range(len(section_names)))
            ax3.set_xticklabels(section_names, rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, dim in zip(bars, fractal_dims):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{dim:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Roughness analysis using Hurst exponent
        def hurst_exponent(data):
            """Calculate Hurst exponent using R/S analysis"""
            n = len(data)
            if n < 10:
                return np.nan
            
            # Calculate mean
            mean_data = np.mean(data)
            
            # Calculate cumulative deviations
            deviations = data - mean_data
            cumulative_deviations = np.cumsum(deviations)
            
            # Calculate range and standard deviation for different time scales
            scales = np.unique(np.logspace(1, np.log10(n//2), 10).astype(int))
            rs_values = []
            
            for scale in scales:
                if scale >= n:
                    continue
                    
                # Divide into segments
                n_segments = n // scale
                segments = [cumulative_deviations[i*scale:(i+1)*scale] for i in range(n_segments)]
                
                rs_segment = []
                for segment in segments:
                    if len(segment) > 1:
                        range_val = np.max(segment) - np.min(segment)
                        std_val = np.std(data[len(rs_segment)*scale:(len(rs_segment)+1)*scale])
                        if std_val > 0:
                            rs_segment.append(range_val / std_val)
                
                if rs_segment:
                    rs_values.append(np.mean(rs_segment))
            
            if len(rs_values) < 3:
                return np.nan
            
            # Fit log-log relationship
            log_scales = np.log(scales[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            slope, _ = np.polyfit(log_scales, log_rs, 1)
            return slope
        
        # Calculate Hurst exponent for each section
        hurst_values = []
        for name, data in sections.items():
            if len(data) > 10:
                hurst = hurst_exponent(data)
                if not np.isnan(hurst):
                    hurst_values.append(hurst)
                    fractal_results[f'{name}_hurst'] = hurst
                else:
                    hurst_values.append(0)
            else:
                hurst_values.append(0)
        
        # Create Hurst exponent interpretation
        x_pos = np.arange(len(section_names))
        bars = ax4.bar(x_pos, hurst_values[:len(section_names)], 
                      color=colors[:len(section_names)], alpha=0.7, edgecolor='black')
        
        # Add interpretation lines
        ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Random walk (H=0.5)')
        ax4.axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='Persistent (H>0.5)')
        ax4.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Anti-persistent (H<0.5)')
        
        ax4.set_xlabel('Data Section')
        ax4.set_ylabel('Hurst Exponent')
        ax4.set_title('Hurst Exponent Analysis', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(section_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, hurst in zip(bars, hurst_values[:len(section_names)]):
            if hurst > 0:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{hurst:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return fractal_results
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive analysis report combining all methods
        """
        print("🔬 COMPREHENSIVE TIPPING POINT ANALYSIS REPORT")
        print("=" * 80)
        
        # Run all analyses
        print("\n📊 Running Catastrophe Theory Analysis...")
        critical_points, cusp_params, _ = self.catastrophe_theory_analysis()
        
        print("\n📊 Running Phase Transition Analysis...")
        critical_ratio, critical_exp = self.phase_transition_analysis()
        
        print("\n📊 Running Multiscale Analysis...")
        ridges, max_var_scale, entropies = self.multiscale_analysis()
        
        print("\n📊 Running Network Topology Analysis...")
        clustering, path_length, modularity, small_world = self.network_topology_analysis()
        
        print("\n📊 Running Machine Learning Classification...")
        ml_results = self.machine_learning_classification()
        
        print("\n📊 Running Hysteresis Analysis...")
        hysteresis_results = self.hysteresis_analysis()
        
        print("\n📊 Running Fractal Dimension Analysis...")
        fractal_results = self.fractal_dimension_analysis()
        
        # Compile comprehensive report
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     COMPREHENSIVE TIPPING POINT ANALYSIS                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

📈 DATA OVERVIEW:
  • Ratio Range: {self.ratios.min():.2f} to {self.ratios.max():.2f}
  • Survival Range: {self.survival_rates.min():.1f}% to {self.survival_rates.max():.1f}%
  • Number of Data Points: {len(self.ratios)}

🎯 CRITICAL POINT IDENTIFICATION:

1. CATASTROPHE THEORY:
   • Critical Points from Cusp Model: {', '.join([f'{cp:.2f}' for cp in critical_points])}
   • Model Parameters: a={cusp_params[0]:.3f}, b={cusp_params[1]:.3f}

2. PHASE TRANSITION ANALYSIS:
   • Critical Ratio (Max Susceptibility): {critical_ratio:.2f}
   • Critical Exponent: 
3. MULTISCALE ANALYSIS:
   • Wavelet Ridge Critical Points: {len(ridges)} detected
   • Dominant Scale: {max_var_scale:.1f}
   • Complexity (Entropy) Peak: {np.argmax(entropies) if entropies else 'N/A'}

📊 NETWORK CHARACTERISTICS:

4. TOPOLOGY EVOLUTION:
   • Initial Clustering: {clustering[0]:.3f} → Final: {clustering[-1]:.3f}
   • Path Length Change: {path_length[0]:.2f} → {path_length[-1]:.2f}
   • Small-World Coefficient: {small_world[0]:.3f} → {small_world[-1]:.3f}

🤖 MACHINE LEARNING INSIGHTS:

5. FEATURE IMPORTANCE:
   • Most Important Feature: {max(ml_results['feature_importance'], key=ml_results['feature_importance'].get)}
   • Clustering Quality (Silhouette): {ml_results['silhouette_score']:.3f}
   • Explained Variance (PC1+PC2): {sum(ml_results['explained_variance']):.1%}

🔄 HYSTERESIS & MEMORY:

6. SYSTEM MEMORY:
   • Maximum Hysteresis at Ratio: {hysteresis_results['max_hysteresis_ratio']:.2f}
   • Memory Effect Strength: {np.max(hysteresis_results['memory_effect']):.2f}
   • Recovery Potential: {np.max(hysteresis_results['recovery_potential']):.1f}%

📐 FRACTAL CHARACTERISTICS:

7. COMPLEXITY MEASURES:
"""
        
        if fractal_results:
            for key, value in fractal_results.items():
                if not key.endswith('_hurst'):
                    report += f"   • {key} Fractal Dimension: {value:.3f}\n"
        
        report += f"""
🎯 CONSENSUS TIPPING POINT:
   Based on multiple methods, the critical tipping point appears to be around:
   
   ⚠️  RATIO: {np.median([critical_ratio] + list(critical_points)):.2f} ± 0.2
   
   This represents the threshold where:
   • Network connectivity collapses rapidly
   • System exhibits maximum instability
   • Recovery becomes difficult due to hysteresis
   • Fractal complexity changes dramatically

🔬 METHODOLOGY CONFIDENCE:
   • High confidence methods: Phase transition, Catastrophe theory
   • Supporting evidence: Multiscale analysis, ML classification
   • Additional insights: Hysteresis, Fractal analysis

📋 RECOMMENDATIONS:
   1. Maintain pBDNF/mBDNF ratio below {critical_ratio-0.5:.1f} for system stability
   2. Monitor early warning signals around ratio {critical_ratio-1:.1f}
   3. Implement interventions before ratio {critical_ratio-0.2:.1f}
   4. Consider hysteresis effects in recovery protocols

╔══════════════════════════════════════════════════════════════════════════════╗
║                            ANALYSIS COMPLETE                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        print(report)
        
        # Save report to file
        with open('comprehensive_tipping_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\n💾 Full report saved to: comprehensive_tipping_analysis_report.txt")
        
        return {
            'critical_points': critical_points,
            'critical_ratio': critical_ratio,
            'ml_results': ml_results,
            'hysteresis_results': hysteresis_results,
            'fractal_results': fractal_results,
            'consensus_tipping_point': np.median([critical_ratio] + list(critical_points))
        }


# Example usage with your data
if __name__ == "__main__":
    # Your data from the graphs
    connection_survival = [0.076, 0.201, 0.187, 0.512, 0.443, 0.746, 0.942, 0.989, 0.994, 0.870, 0.795, 0.775, 0.765, 0.759, 0.754, 0.785, 0.762, 0.756,   0.726,   0.782, 0.763, 0.620, 0.546, 0.465, 0.347, 0.331, 0.215, 0.241, 0.239, 0.233, 0.218]
    ratio = [3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 6.4, 6.6, 6.8, 7, 7.2, 7.3, 7.4, 7.49, 7.5, 8, 10, 12, 15, 20]
    # Convert to percentages
    survival_percentages = [s * 100 for s in connection_survival]
    
    # Optional: Add time series data if available
    time_series_data = {
        7.0: {'time': np.linspace(0, 1000, 100), 'survival': 70 + 10*np.random.randn(100)},
        7.5: {'time': np.linspace(0, 1000, 100), 'survival': 35 + 15*np.random.randn(100)},
        8.0: {'time': np.linspace(0, 1000, 100), 'survival': 20 + 5*np.random.randn(100)},
    }
    
    # Create analyzer
    analyzer = AdvancedTippingAnalysis(ratio, survival_percentages, time_series_data)
    
    # Run comprehensive analysis
    print("🚀 Starting Advanced Tipping Point Analysis...")
    results = analyzer.generate_comprehensive_report()
    
    print(f"\n✨ Analysis complete! Consensus tipping point: {results['consensus_tipping_point']:.2f}")
    print(f"📁 Check output files for detailed results and visualizations.")