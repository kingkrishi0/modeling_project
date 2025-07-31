# Define specific test protocols
import os
import numpy as np
import matplotlib.pyplot as plt
from neuron import h

from biologically_accurate import MinimalBiologicalNetwork  
from ode_neuron_class import ODENeuron


def run_bdnf_parameter_test(network_class, base_params, base_concentrations, 
                           test_name, parameter_modifications, simulation_time=500,
                           visualize_networks=True, visualization_times=[0, 250, 500]):
    """
    Run a systematic test of how parameter changes affect BDNF/proBDNF balance
    
    Args:
        network_class: Your MinimalBiologicalNetwork class
        base_params: Base parameter set
        base_concentrations: Base initial concentrations
        test_name: Name for this test
        parameter_modifications: Dict of parameter changes to test
        simulation_time: How long to run each test
        visualize_networks: Whether to save network visualizations
        visualization_times: Times at which to save network visualizations
    
    Returns:
        results: Dict containing all test results
    """
    
    results = {
        'test_name': test_name,
        'conditions': [],
        'final_metrics': [],
        'bdnf_timeseries': [],
        'probdnf_timeseries': [],
        'health_timeseries': [],
        'layer_analysis': [],
        'networks': []  # Store network objects for visualization
    }
    
    # Create main results directory
    main_save_path = f"bdnf_tests/{test_name}"
    os.makedirs(main_save_path, exist_ok=True)
    
    print(f"\n🧪 Starting {test_name}")
    print("="*60)
    
    for condition_idx, (condition_name, param_changes) in enumerate(parameter_modifications.items()):
        print(f"\n🔬 Testing condition: {condition_name}")
        
        # Create condition-specific directory
        condition_save_path = f"{main_save_path}/{condition_name}"
        os.makedirs(condition_save_path, exist_ok=True)
        
        # Modify parameters for this condition
        test_params = base_params.copy()
        for param_idx, multiplier in param_changes.items():
            test_params[param_idx] *= multiplier
            param_names = [
                "ksP", "k_cleave", "k_p75_pro_on", "k_p75_pro_off", "k_degP", 
                "k_TrkB_pro_on", "k_TrkB_pro_off", "k_TrkB_B_on", "k_TrkB_B_off", 
                "k_degB", "k_p75_B_on", "k_p75_B_off", "k_degR1", "k_degR2",
                "k_int_p75_pro", "k_int_p75_B", "k_int_TrkB_B", "k_int_TrkB_pro", 
                "aff_p75_pro", "aff_p75_B", "aff_TrkB_pro", "aff_TrkB_B", 
                "k_deg_tPA", "ks_tPA", "ks_p75", "ks_TrkB", "tau_activity", "activity_gain"
            ]
            if param_idx < len(param_names):
                print(f"  Modified {param_names[param_idx]} by {multiplier}x")
        
        # Create network with modified parameters
        network = network_class(
            rows=5, cols=6,
            initial_neuron_concentrations=base_concentrations,
            base_neuron_params=test_params,
            synapse_type="ProbabilisticSyn",
            initial_syn_weight=0.6,
            learning_rate=0.002
        )
        
        # Run simulation with visualization
        h.finitialize(-65)
        
        # Data collection
        time_points = []
        bdnf_values = []
        probdnf_values = []
        health_values = []
        
        current_time = 0.0
        time_step = 0
        last_update = 0.0
        last_visualization = -100.0
        
        # Save initial network state
        if visualize_networks and 0 in visualization_times:
            print(f"    📸 Saving network visualization at t=0ms")
            plt.figure(figsize=(12, 10))
            network.visualize_bdnf_network(0, 0, save_fig=False)
            plt.suptitle(f"{condition_name} - Network at t=0ms", fontsize=14)
            plt.savefig(f"{condition_save_path}/network_t000.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # ...existing setup code...

        visualized_times = set()

        while current_time <= simulation_time:
            h.fadvance()
            current_time = h.t
            time_step += 1

            # Apply plasticity
            if current_time - last_update >= 10.0:
                network.bdnf_driven_plasticity()
                last_update = current_time

            # Network visualization at specific times (limit to 5)
            if visualize_networks:
                for viz_time in visualization_times:
                    if (viz_time not in visualized_times) and (abs(current_time - viz_time) < 1.0):
                        print(f"    📸 Saving network visualization at t={current_time:.0f}ms")
                        plt.figure(figsize=(12, 10))
                        network.visualize_bdnf_network(time_step, current_time, save_fig=False)
                        plt.suptitle(f"{condition_name} - Network at t={current_time:.0f}ms", fontsize=14)
                        plt.savefig(f"{condition_save_path}/network_t{int(current_time):03d}.png", dpi=150, bbox_inches='tight')
                        plt.close()
                        visualized_times.add(viz_time)

            # Collect data every 25ms
            if int(current_time) % 25 == 0:
                metrics = network.get_network_health_metrics()
                time_points.append(current_time)
                bdnf_values.append(metrics['avg_bdnf'])
                probdnf_values.append(metrics['avg_probdnf'])
                health_values.append(metrics['network_health'])

        # Final network visualization
        if visualize_networks:
            print(f"    📸 Saving final network visualization")
            plt.figure(figsize=(12, 10))
            network.visualize_bdnf_network(time_step, current_time, save_fig=False)
            plt.suptitle(f"{condition_name} - Final Network State", fontsize=14)
            plt.savefig(f"{condition_save_path}/network_final.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # Create individual condition plot
        plot_individual_condition(condition_name, time_points, bdnf_values, probdnf_values, 
                                health_values, condition_save_path)
        
        # Final analysis
        final_metrics = network.get_network_health_metrics()
        
        # Layer-wise analysis
        layer_stats = {}
        for r in range(network.rows):
            for c in range(network.cols):
                neuron = network.neurons[r][c]
                layer = neuron.cortical_layer
                
                if layer not in layer_stats:
                    layer_stats[layer] = {'bdnf': [], 'probdnf': [], 'health': []}
                
                layer_stats[layer]['bdnf'].append(neuron.ode_mech.B)
                layer_stats[layer]['probdnf'].append(neuron.ode_mech.P)
                layer_stats[layer]['health'].append(neuron.calculate_and_get_neuron_state())
        
        # Average layer stats
        for layer in layer_stats:
            layer_stats[layer] = {
                'avg_bdnf': np.mean(layer_stats[layer]['bdnf']),
                'avg_probdnf': np.mean(layer_stats[layer]['probdnf']),
                'avg_health': np.mean(layer_stats[layer]['health'])
            }
        
        # Store results
        results['conditions'].append(condition_name)
        results['final_metrics'].append(final_metrics)
        results['bdnf_timeseries'].append(bdnf_values)
        results['probdnf_timeseries'].append(probdnf_values)
        results['health_timeseries'].append(health_values)
        results['layer_analysis'].append(layer_stats)
        results['networks'].append(network)  # Store for later analysis
        
        print(f"  Final BDNF: {final_metrics['avg_bdnf']:.3f}")
        print(f"  Final proBDNF: {final_metrics['avg_probdnf']:.3f}")
        print(f"  Final Health: {final_metrics['network_health']:.3f}")
        print(f"  Active Connections: {final_metrics['active_connections']}")
        print(f"  📁 Results saved to: {condition_save_path}")
    
    return results

def plot_individual_condition(condition_name, time_points, bdnf_values, probdnf_values, 
                             health_values, save_path):
    """Plot detailed analysis for individual condition"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{condition_name} - Detailed BDNF Analysis", fontsize=16)
    
    times = np.array(time_points)
    
    # Plot 1: BDNF vs proBDNF over time
    ax1 = axes[0, 0]
    ax1.plot(times, bdnf_values, 'g-', linewidth=2, label='BDNF', marker='o', markersize=3)
    ax1.plot(times, probdnf_values, 'r-', linewidth=2, label='proBDNF', marker='o', markersize=3)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Concentration (μM)')
    ax1.set_title('BDNF vs proBDNF Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Network health over time
    ax2 = axes[0, 1]
    ax2.plot(times, health_values, 'b-', linewidth=2, marker='o', markersize=3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Network Health')
    ax2.set_title('Network Health Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: BDNF/proBDNF ratio over time
    ax3 = axes[1, 0]
    bdnf_array = np.array(bdnf_values)
    probdnf_array = np.array(probdnf_values)
    ratio = bdnf_array / np.maximum(probdnf_array, 0.001)  # Avoid division by zero
    ax3.plot(times, ratio, 'purple', linewidth=2, marker='o', markersize=3)
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Balanced')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('BDNF/proBDNF Ratio')
    ax3.set_title('BDNF/proBDNF Balance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Rate of change
    ax4 = axes[1, 1]
    if len(health_values) > 1:
        health_change = np.diff(health_values)
        times_diff = times[1:]
        ax4.plot(times_diff, health_change, 'orange', linewidth=2, marker='o', markersize=3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Health Change Rate')
        ax4.set_title('Network Health Change Rate')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/detailed_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close to prevent display

def plot_bdnf_test_results(results, save_path="bdnf_tests"):
    """Plot comprehensive results from BDNF parameter tests"""
    
    os.makedirs(save_path, exist_ok=True)
    
    n_conditions = len(results['conditions'])
    colors = plt.cm.Set3(np.linspace(0, 1, n_conditions))
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{results['test_name']} - BDNF/proBDNF Parameter Analysis", fontsize=16)
    
    # Plot 1: BDNF evolution over time
    ax1 = axes[0, 0]
    for i, condition in enumerate(results['conditions']):
        time_points = np.arange(0, len(results['bdnf_timeseries'][i])) * 25
        ax1.plot(time_points, results['bdnf_timeseries'][i], 
                color=colors[i], linewidth=2, label=condition)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Average BDNF (μM)')
    ax1.set_title('BDNF Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: proBDNF evolution over time
    ax2 = axes[0, 1]
    for i, condition in enumerate(results['conditions']):
        time_points = np.arange(0, len(results['probdnf_timeseries'][i])) * 25
        ax2.plot(time_points, results['probdnf_timeseries'][i], 
                color=colors[i], linewidth=2, label=condition)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Average proBDNF (μM)')
    ax2.set_title('proBDNF Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Network health evolution
    ax3 = axes[0, 2]
    for i, condition in enumerate(results['conditions']):
        time_points = np.arange(0, len(results['health_timeseries'][i])) * 25
        ax3.plot(time_points, results['health_timeseries'][i], 
                color=colors[i], linewidth=2, label=condition)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Network Health')
    ax3.set_title('Network Health Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Final BDNF/proBDNF comparison
    ax4 = axes[1, 0]
    final_bdnf = [metrics['avg_bdnf'] for metrics in results['final_metrics']]
    final_probdnf = [metrics['avg_probdnf'] for metrics in results['final_metrics']]
    
    x_pos = np.arange(len(results['conditions']))
    width = 0.35
    
    ax4.bar(x_pos - width/2, final_bdnf, width, label='BDNF', color='green', alpha=0.7)
    ax4.bar(x_pos + width/2, final_probdnf, width, label='proBDNF', color='red', alpha=0.7)
    ax4.set_xlabel('Condition')
    ax4.set_ylabel('Final Concentration (μM)')
    ax4.set_title('Final BDNF vs proBDNF')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(results['conditions'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Layer 5 vulnerability analysis
    ax5 = axes[1, 1]
    layer5_bdnf = []
    layer5_probdnf = []
    layer5_health = []
    
    for layer_stats in results['layer_analysis']:
        if 'Layer5' in layer_stats:
            layer5_bdnf.append(layer_stats['Layer5']['avg_bdnf'])
            layer5_probdnf.append(layer_stats['Layer5']['avg_probdnf'])
            layer5_health.append(layer_stats['Layer5']['avg_health'])
        else:
            layer5_bdnf.append(0)
            layer5_probdnf.append(0)
            layer5_health.append(0)
    
    ax5.bar(x_pos - width/2, layer5_bdnf, width, label='BDNF', color='green', alpha=0.7)
    ax5.bar(x_pos + width/2, layer5_probdnf, width, label='proBDNF', color='red', alpha=0.7)
    ax5.set_xlabel('Condition')
    ax5.set_ylabel('Layer 5 Concentration (μM)')
    ax5.set_title('Layer 5 BDNF/proBDNF (Most Vulnerable)')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(results['conditions'], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Network connectivity
    ax6 = axes[1, 2]
    final_connections = [metrics['active_connections'] for metrics in results['final_metrics']]
    initial_connections = [137] * len(results['conditions'])  # Assuming baseline
    
    ax6.bar(x_pos - width/2, initial_connections, width, label='Initial', color='blue', alpha=0.7)
    ax6.bar(x_pos + width/2, final_connections, width, label='Final', color='orange', alpha=0.7)
    ax6.set_xlabel('Condition')
    ax6.set_ylabel('Active Connections')
    ax6.set_title('Network Connectivity Changes')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(results['conditions'], rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/{results["test_name"]}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n📊 {results['test_name']} SUMMARY:")
    print("="*50)
    for i, condition in enumerate(results['conditions']):
        metrics = results['final_metrics'][i]
        print(f"\n{condition}:")
        print(f"  BDNF: {metrics['avg_bdnf']:.3f} μM")
        print(f"  proBDNF: {metrics['avg_probdnf']:.3f} μM")
        print(f"  Health: {metrics['network_health']:.3f}")
        print(f"  Connections: {metrics['active_connections']}")
        print(f"  BDNF/proBDNF Ratio: {metrics['avg_bdnf']/max(metrics['avg_probdnf'], 0.001):.3f}")

# Define specific test protocols

def test_activity_effects(network_class, base_params, base_concentrations):
    """Test how activity levels affect BDNF/proBDNF balance"""
    
    activity_modifications = {
        "Low_Activity": {27: 0.5},      # Reduce activity_gain
        "Baseline": {},                  # No changes
        "High_Activity": {27: 2.0},     # Increase activity_gain
        "Very_High_Activity": {27: 5.0} # Very high activity_gain
    }
    
    return run_bdnf_parameter_test(
        network_class, base_params, base_concentrations,
        "Activity_Effects_on_BDNF", activity_modifications,
        visualize_networks=True, visualization_times=[0, 125, 250, 375, 500]
    )

def test_synthesis_rates(network_class, base_params, base_concentrations):
    """Test how BDNF synthesis rates affect network health"""
    
    synthesis_modifications = {
        "Low_proBDNF_Synthesis": {0: 0.5},    # Reduce ksP
        "Baseline": {},
        "High_proBDNF_Synthesis": {0: 2.0},   # Increase ksP
        "Low_tPA": {23: 0.5},                 # Reduce tPA synthesis (less BDNF processing)
        "High_tPA": {23: 2.0},                # Increase tPA synthesis (more BDNF processing)
    }
    
    return run_bdnf_parameter_test(
        network_class, base_params, base_concentrations,
        "BDNF_Synthesis_Effects", synthesis_modifications,
        visualize_networks=True, visualization_times=[0, 125, 250, 375, 500]
    )

def test_degradation_rates(network_class, base_params, base_concentrations):
    """Test how BDNF/proBDNF degradation affects network"""
    
    degradation_modifications = {
        "Slow_BDNF_Degradation": {9: 0.5},    # Slower BDNF degradation
        "Fast_BDNF_Degradation": {9: 2.0},    # Faster BDNF degradation
        "Slow_proBDNF_Degradation": {4: 0.5}, # Slower proBDNF degradation
        "Fast_proBDNF_Degradation": {4: 2.0}, # Faster proBDNF degradation
        "Baseline": {}
    }
    
    return run_bdnf_parameter_test(
        network_class, base_params, base_concentrations,
        "Degradation_Effects", degradation_modifications,
        visualize_networks=True, visualization_times=[0, 125, 250, 375, 500]
    )

def test_receptor_balance(network_class, base_params, base_concentrations):
    """Test how TrkB/p75 receptor balance affects network health"""
    
    receptor_modifications = {
        "High_TrkB": {25: 2.0},           # More TrkB receptors (survival)
        "Low_TrkB": {25: 0.5},            # Fewer TrkB receptors
        "High_p75": {24: 2.0},            # More p75 receptors (death)
        "Low_p75": {24: 0.5},             # Fewer p75 receptors
        "TrkB_Dominant": {25: 2.0, 24: 0.5},  # Favor survival
        "p75_Dominant": {25: 0.5, 24: 2.0},   # Favor death
        "Baseline": {}
    }
    
    return run_bdnf_parameter_test(
        network_class, base_params, base_concentrations,
        "Receptor_Balance_Effects", receptor_modifications,
        visualize_networks=True, visualization_times=[0, 125, 250, 375, 500]
    )

def create_comparison_gif(test_results, condition_name, save_path):
    """Create animated GIF showing network evolution for a specific condition"""
    try:
        from PIL import Image
        import glob
        
        # Find all network images for this condition
        condition_path = f"{save_path}/{test_results['test_name']}/{condition_name}"
        image_files = sorted(glob.glob(f"{condition_path}/network_t*.png"))
        
        if len(image_files) > 1:
            # Load images
            images = []
            for img_file in image_files:
                img = Image.open(img_file)
                images.append(img)
            
            # Save as GIF
            gif_path = f"{condition_path}/network_evolution.gif"
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=1000,  # 1 second per frame
                loop=0
            )
            print(f"    🎬 Created evolution GIF: {gif_path}")
        
    except ImportError:
        print("    ⚠️  PIL not available - skipping GIF creation")
    except Exception as e:
        print(f"    ⚠️  GIF creation failed: {e}")

def create_comprehensive_comparison_plots(results, save_path="bdnf_tests"):
    """Create detailed comparison plots for all conditions"""
    
    test_name = results['test_name']
    main_path = f"{save_path}/{test_name}"
    os.makedirs(main_path, exist_ok=True)
    
    # Plot 1: Side-by-side network evolution comparison
    n_conditions = len(results['conditions'])
    n_timepoints = 5  # Initial, 25%, 50%, 75%, Final
    
    fig, axes = plt.subplots(n_conditions, n_timepoints, figsize=(20, 4*n_conditions))
    if n_conditions == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"{test_name} - Network Evolution Comparison", fontsize=16)
    
    # This would require storing network states at different times
    # For now, we'll create the time series comparison
    
    # Plot 2: Detailed time series comparison
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle(f"{test_name} - Detailed Time Series Analysis", fontsize=16)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_conditions))
    
    # BDNF evolution
    ax1 = axes2[0, 0]
    for i, condition in enumerate(results['conditions']):
        times = np.arange(0, len(results['bdnf_timeseries'][i])) * 25
        ax1.plot(times, results['bdnf_timeseries'][i], 
                color=colors[i], linewidth=3, label=condition, marker='o', markersize=4)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Average BDNF (μM)')
    ax1.set_title('BDNF Evolution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # proBDNF evolution
    ax2 = axes2[0, 1]
    for i, condition in enumerate(results['conditions']):
        times = np.arange(0, len(results['probdnf_timeseries'][i])) * 25
        ax2.plot(times, results['probdnf_timeseries'][i], 
                color=colors[i], linewidth=3, label=condition, marker='s', markersize=4)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Average proBDNF (μM)')
    ax2.set_title('proBDNF Evolution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Health evolution
    ax3 = axes2[1, 0]
    for i, condition in enumerate(results['conditions']):
        times = np.arange(0, len(results['health_timeseries'][i])) * 25
        ax3.plot(times, results['health_timeseries'][i], 
                color=colors[i], linewidth=3, label=condition, marker='^', markersize=4)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Network Health')
    ax3.set_title('Network Health Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # BDNF/proBDNF ratio evolution
    ax4 = axes2[1, 1]
    for i, condition in enumerate(results['conditions']):
        times = np.arange(0, len(results['bdnf_timeseries'][i])) * 25
        bdnf_array = np.array(results['bdnf_timeseries'][i])
        probdnf_array = np.array(results['probdnf_timeseries'][i])
        ratio = bdnf_array / np.maximum(probdnf_array, 0.001)
        ax4.plot(times, ratio, color=colors[i], linewidth=3, 
                label=condition, marker='d', markersize=4)
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Balanced')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('BDNF/proBDNF Ratio')
    ax4.set_title('BDNF/proBDNF Balance Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{main_path}/{test_name}_detailed_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create GIFs for each condition
    for condition in results['conditions']:
        create_comparison_gif(results, condition, save_path)
    
    print(f"\n📊 Comprehensive analysis saved to: {main_path}")
    print("📁 Individual condition results in subfolders")
    print("🎬 Network evolution GIFs created (if PIL available)")

# Example usage
if __name__ == "__main__":
    """
    🧪 Enhanced BDNF/proBDNF Parameter Testing Framework
    
    Now includes:
    ✓ Individual network visualizations for each condition
    ✓ Network evolution over time (multiple timepoints)
    ✓ Detailed analysis plots for each condition
    ✓ Animated GIFs showing network evolution
    ✓ Comprehensive comparison plots
    
    📊 Available Tests:
    1. test_activity_effects() - How activity levels affect BDNF balance
    2. test_synthesis_rates() - How synthesis rates affect network health  
    3. test_degradation_rates() - How degradation affects BDNF/proBDNF
    4. test_receptor_balance() - How TrkB/p75 balance affects survival
    
    🔬 Enhanced Usage:
    """
    
    # Load mechanisms first
    mod_dir = os.path.join(os.path.dirname(__file__), 'mods')
    if os.path.exists(mod_dir):
        os.system(f"nrnivmodl {mod_dir}")
    else:
        os.system("nrnivmodl")
    
    try:
        lib_paths = [
            os.path.join("x86_64", ".libs", "libnrnmech.so"),
            os.path.join("arm64", ".libs", "libnrnmech.dylib"),
            "nrnmech.dll"
        ]
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                try:
                    h.nrn_load_dll(lib_path)
                    print(f"✓ Loaded mechanisms from {lib_path}")
                    break
                except:
                    continue
    except Exception as e:
        print(f"Error loading mechanisms: {e}")
    
    # Parameters
    initial_neuron_concentrations = [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    base_neuron_parameters = [
        5.0e-3, 0.01, 1.0, 0.9, 5.0e-4, 0.2, 0.1, 1.0, 0.9, 0.005, 0.3, 0.1, 
        0.0001, 0.00001, 0.0005, 0.0005, 0.0005, 0.0005, 0.9, 0.1, 0.1, 0.9, 
        0.0011, 0.001, 0.002, 0.003, 0.004, 1.0, 1.0, 0.0001, -65.0, -20.0
    ]
    
    print("\n🧪 Running Enhanced BDNF Parameter Tests")
    print("="*60)
    
    # Test 1: Activity Effects
    print("\n🔬 Test 1: Activity Effects on BDNF Balance")
    activity_results = test_activity_effects(
        MinimalBiologicalNetwork, base_neuron_parameters, initial_neuron_concentrations
    )
    
    # Create comprehensive comparison
    create_comprehensive_comparison_plots(activity_results)
    
    # Test 2: Synthesis Rates (optional - uncomment to run)
    # print("\n🔬 Test 2: BDNF Synthesis Rate Effects")
    # synthesis_results = test_synthesis_rates(
    #     MinimalBiologicalNetwork, base_neuron_parameters, initial_neuron_concentrations
    # )
    # create_comprehensive_comparison_plots(synthesis_results)
    
    print("\n✨ Enhanced BDNF Testing Complete!")
    print("""
    📁 Results Structure:
    bdnf_tests/
    ├── Activity_Effects_on_BDNF/
    │   ├── Low_Activity/
    │   │   ├── network_t000.png
    │   │   ├── network_t125.png
    │   │   ├── network_t250.png
    │   │   ├── network_t375.png
    │   │   ├── network_final.png
    │   │   ├── network_evolution.gif
    │   │   └── detailed_analysis.png
    │   ├── Baseline/
    │   ├── High_Activity/
    │   ├── Very_High_Activity/
    │   └── Activity_Effects_on_BDNF_detailed_comparison.png
    
    🎯 What You Get:
    ✓ Network visualizations at multiple timepoints for each condition
    ✓ Animated GIFs showing network evolution
    ✓ Individual detailed analysis for each condition
    ✓ Comprehensive comparison plots across all conditions
    ✓ Layer-specific BDNF/proBDNF analysis
    ✓ Time series data for all metrics
    """)