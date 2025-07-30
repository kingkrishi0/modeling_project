from neuron import h, nrn
from neuronpp.cells.cell import Cell
from neuronpp.core.hocwrappers.synapses.single_synapse import SingleSynapse
import numpy as np
from neuronpp.core.hocwrappers.point_process import PointProcess
import os
from ode_neuron_class import ODENeuron
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CorrectedNeuronalNetwork:
    def __init__(self, rows: int, cols: int, initial_neuron_concentrations: list, neuron_params: list,
                 synapse_type: str = "ProbabilisticSyn",
                 initial_syn_weight: float = 0.5,
                 learning_rate: float = 0.001,
                 min_weight: float = 0.05,  # Prevent complete decay
                 max_weight: float = 1.0,
                 threshold_growth: float = 0.1,  # Lower thresholds
                 threshold_apoptosis: float = -0.1):
        
        if rows<=0 or cols<=0:
            raise ValueError("rows and cols must be above 0")
        
        self.rows = rows
        self.cols = cols
        self.num_neurons = rows * cols
        self.neurons = []
        self.connections = []

        self._created_unidirectional_connections = set()
        self.synapse_type = synapse_type
        self.initial_syn_weight = initial_syn_weight
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.threshold_growth = threshold_growth
        self.threshold_apoptosis = threshold_apoptosis

        print(f"Building corrected neuronal network ({rows}x{cols})")

        # Create neurons
        for r in range(self.rows):
            newrow = []
            for c in range(self.cols):
                neuron = ODENeuron(name=f'neuron_r{r}c{c}', 
                                 initial_concentrations=initial_neuron_concentrations, 
                                 params=neuron_params)
                newrow.append(neuron)
            self.neurons.append(newrow)

        self._build_grid_connections()
        print(f"Network created with {len(self.connections)} connections!")

    def _get_neuron_at_grid_pos(self, r: int, c: int):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.neurons[r][c]
        return None
    
    def _create_connection(self, pre_neuron, post_neuron, pre_r, pre_c, post_r, post_c):
        conn_id = f"{pre_neuron.name}_to_{post_neuron.name}"

        if conn_id in self._created_unidirectional_connections:
            return False
        
        try:
            post_segment_loc = post_neuron.soma(0.5) 
            h_section_at_segment = (post_segment_loc.parent.hoc)(post_segment_loc.x)

            if hasattr(h, self.synapse_type):
                raw_h_point_process = getattr(h, self.synapse_type)(h_section_at_segment)
            else:
                raise ValueError(f"Synapse type '{self.synapse_type}' not found")

            point_process_mech = PointProcess(
                hoc_obj=raw_h_point_process,
                parent=post_segment_loc.parent,
                name=f"{self.synapse_type}_{conn_id}",
                cell=post_neuron,
                mod_name=self.synapse_type
            )
            
            # Create NetCon - FIXED: Use membrane voltage directly
            netcon_to_synapse = h.NetCon(pre_neuron.soma_segment.hoc._ref_v, 
                                       point_process_mech.hoc, 
                                       sec=pre_neuron.soma.hoc)
            netcon_to_synapse.weight[0] = self.initial_syn_weight
            netcon_to_synapse.delay = 1.0
            netcon_to_synapse.threshold = -25.0  # More sensitive threshold

            # CRITICAL FIX: Set the pointer correctly
            try:
                h.setpointer(post_neuron.ode_mech._ref_syn_input_activity, 
                           'target_syn_input_activity', 
                           point_process_mech.hoc)
                print(f"✓ Pointer set for {conn_id}")
            except Exception as e:
                print(f"✗ Pointer failed for {conn_id}: {e}")
                return False
            
            syn_obj = SingleSynapse(
                source=pre_neuron.spike_detector,
                point_process=point_process_mech, 
                name=conn_id 
            )
            syn_obj.netcons.append(netcon_to_synapse)

            connection_info = {
                'pre_neuron': pre_neuron,
                'post_neuron': post_neuron,
                'syn': syn_obj,
                'netcon_to_synapse': netcon_to_synapse,
                'pre_r': pre_r,
                'pre_c': pre_c,
                'post_r': post_r,
                'post_c': post_c
            }
            
            self.connections.append(connection_info)
            self._created_unidirectional_connections.add(conn_id)
            
            print(f"✓ Connection: {pre_neuron.name} -> {post_neuron.name} (weight={netcon_to_synapse.weight[0]:.3f})")
            return True
            
        except Exception as e:
            print(f"✗ Failed to create connection {conn_id}: {e}")
            return False

    def _build_grid_connections(self):
        if self.num_neurons < 2:
            print("Need at least 2 neurons")
            return

        connection_count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                pre_neuron = self._get_neuron_at_grid_pos(r, c)
                
                # Connect to right and down neighbors (bidirectionally)
                offsets = [(0,1), (1, 0)]

                for dr, dc in offsets:
                    post_r, post_c = r+dr, c+dc
                    post_neuron = self._get_neuron_at_grid_pos(post_r, post_c)

                    if post_neuron:
                        # Create both directions
                        if self._create_connection(pre_neuron, post_neuron, r, c, post_r, post_c):
                            connection_count += 1
                        if self._create_connection(post_neuron, pre_neuron, post_r, post_c, r, c):
                            connection_count += 1
        
        print(f"Total connections created: {connection_count}")

    def modulate_synaptic_weights(self):
        """Improved synaptic weight modulation"""
        for conn_info in self.connections:
            pre_neuron = conn_info['pre_neuron']
            post_neuron = conn_info['post_neuron']
            netcon_to_synapse = conn_info['netcon_to_synapse']

            pre_n_state = pre_neuron.calculate_and_get_neuron_state()
            post_n_state = post_neuron.calculate_and_get_neuron_state()

            current_weight = netcon_to_synapse.weight[0]
            weight_change = 0.0

            # Hebbian-like learning with activity correlation
            if pre_n_state > self.threshold_growth and post_n_state > self.threshold_growth:
                # Both neurons in growth state - strengthen connection
                weight_change = self.learning_rate * 2.0
            elif pre_n_state < self.threshold_apoptosis or post_n_state < self.threshold_apoptosis:
                # One or both in apoptosis state - weaken connection
                weight_change = -self.learning_rate * 0.5
            else:
                # Neutral state - small positive drift to maintain baseline connectivity
                weight_change = self.learning_rate * 0.1

            new_weight = current_weight + weight_change
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))

            netcon_to_synapse.weight[0] = new_weight

    def print_network_state(self, time_step, current_time):
        """Print detailed network state"""
        print(f"\n=== Network State at Step {time_step}, Time = {current_time:.1f}ms ===")
        
        for r in range(self.rows):
            for c in range(self.cols):
                neuron = self.neurons[r][c]
                state = neuron.calculate_and_get_neuron_state()
                
                # Get current values safely
                try:
                    activity = neuron.activity_level_ref[0] if len(neuron.activity_level_ref) > 0 else neuron.ode_mech.activity_level
                    syn_input = neuron.syn_input_activity_ref[0] if len(neuron.syn_input_activity_ref) > 0 else neuron.ode_mech.syn_input_activity
                    voltage = neuron.v_ref[0] if len(neuron.v_ref) > 0 else neuron.soma_segment.hoc.v
                    P_conc = neuron.P[0] if len(neuron.P) > 0 else neuron.ode_mech.P
                    B_conc = neuron.B[0] if len(neuron.B) > 0 else neuron.ode_mech.B
                except:
                    activity = neuron.ode_mech.activity_level
                    syn_input = neuron.ode_mech.syn_input_activity
                    voltage = neuron.soma_segment.hoc.v
                    P_conc = neuron.ode_mech.P
                    B_conc = neuron.ode_mech.B
                
                print(f"  {neuron.name}: State={state:.3f}, V={voltage:.1f}mV, "
                      f"SynInput={syn_input:.3f}, Activity={activity:.3f}, "
                      f"P={P_conc:.3f}, B={B_conc:.3f}")
        
        # Print connection weights
        print("  Connection Weights:")
        for conn in self.connections[:4]:  # Show first 4 connections
            weight = conn['netcon_to_synapse'].weight[0]
            pre_name = conn['pre_neuron'].name
            post_name = conn['post_neuron'].name
            print(f"    {pre_name} -> {post_name}: {weight:.4f}")
        if len(self.connections) > 4:
            print(f"    ... and {len(self.connections)-4} more connections")

    def visualize_network(self, time_step, current_time, save_fig=False):
        """Create network visualization"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Calculate neuron positions
        neuron_positions = {}
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * 3.0
                y = (self.rows - r - 1) * 3.0
                neuron_positions[(r, c)] = (x, y)
        
        # Draw connections
        for conn in self.connections:
            if all(k in conn for k in ['pre_r', 'pre_c', 'post_r', 'post_c']):
                pre_pos = neuron_positions[(conn['pre_r'], conn['pre_c'])]
                post_pos = neuron_positions[(conn['post_r'], conn['post_c'])]
                
                weight = conn['netcon_to_synapse'].weight[0]
                thickness = max(0.5, min(8.0, weight * 10))
                
                if weight > 0.5:
                    color = 'darkgreen'
                elif weight > 0.2:
                    color = 'orange'
                else:
                    color = 'red'
                
                # Draw connection line
                ax.plot([pre_pos[0], post_pos[0]], [pre_pos[1], post_pos[1]], 
                       color=color, linewidth=thickness, alpha=0.7)
                
                # Add directional arrow
                dx = post_pos[0] - pre_pos[0]
                dy = post_pos[1] - pre_pos[1]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Arrow at 80% of the way
                    arrow_x = pre_pos[0] + 0.8 * dx
                    arrow_y = pre_pos[1] + 0.8 * dy
                    ax.annotate('', xy=(arrow_x + 0.1*dx, arrow_y + 0.1*dy), 
                               xytext=(arrow_x, arrow_y),
                               arrowprops=dict(arrowstyle='->', color=color, lw=thickness/2))
        
        # Draw neurons
        for r in range(self.rows):
            for c in range(self.cols):
                neuron = self.neurons[r][c]
                pos = neuron_positions[(r, c)]
                
                state = neuron.calculate_and_get_neuron_state()
                try:
                    activity = neuron.activity_level_ref[0] if len(neuron.activity_level_ref) > 0 else neuron.ode_mech.activity_level
                    voltage = neuron.v_ref[0] if len(neuron.v_ref) > 0 else neuron.soma_segment.hoc.v
                except:
                    activity = neuron.ode_mech.activity_level
                    voltage = neuron.soma_segment.hoc.v
                
                # Color based on state
                if state > self.threshold_growth:
                    neuron_color = 'lightgreen'
                elif state < self.threshold_apoptosis:
                    neuron_color = 'lightcoral'
                else:
                    neuron_color = 'lightblue'
                
                # Size based on activity
                size = max(500, min(2000, 500 + abs(activity) * 50))
                
                # Special highlight for spiking neurons
                edge_color = 'red' if voltage > -30 else 'black'
                edge_width = 4 if voltage > -30 else 2
                
                ax.scatter(pos[0], pos[1], s=size, c=neuron_color, 
                          edgecolors=edge_color, linewidth=edge_width, alpha=0.8)
                
                # Label with key info
                ax.text(pos[0], pos[1], f'({r},{c})\nS:{state:.2f}\nV:{voltage:.0f}mV\nA:{activity:.1f}',
                       ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlim(-1, self.cols * 3)
        ax.set_ylim(-1, self.rows * 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Neural Network at t={current_time:.1f}ms (Step {time_step})', fontsize=14)
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='darkgreen', lw=4, label='Strong (>0.5)'),
            plt.Line2D([0], [0], color='orange', lw=4, label='Medium (0.2-0.5)'),
            plt.Line2D([0], [0], color='red', lw=4, label='Weak (<0.2)'),
            plt.scatter([], [], s=500, c='lightgreen', edgecolors='black', label='Growth'),
            plt.scatter([], [], s=500, c='lightcoral', edgecolors='black', label='Apoptosis'),
            plt.scatter([], [], s=500, c='lightblue', edgecolors='black', label='Neutral'),
            plt.scatter([], [], s=500, c='gray', edgecolors='red', linewidth=4, label='Spiking')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'network_t{time_step:04d}.png', dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.1)

# Main execution
if __name__ == "__main__":
    print("=== Starting Corrected Neuronal Network Simulation ===")

    # Compile NMODL files
    mod_dir = os.path.join(os.path.dirname(__file__), 'mods')
    if os.path.exists(mod_dir):
        print(f"Compiling NMODL files in {mod_dir}...")
        os.system(f"nrnivmodl {mod_dir}")
    else:
        print("Compiling NMODL files in current directory...")
        os.system("nrnivmodl")

    # Load mechanisms
    try:
        lib_paths = [
            os.path.join("x86_64", ".libs", "libnrnmech.so"),
            os.path.join("arm64", ".libs", "libnrnmech.dylib"),
            os.path.join("x86_64", ".libs", "libnrnmech.dylib"),
            "nrnmech.dll"
        ]
        
        loaded = False
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                try:
                    h.nrn_load_dll(lib_path)
                    print(f"✓ Loaded mechanisms from {lib_path}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load {lib_path}: {e}")
                    continue
        
        if not loaded:
            print("⚠ Warning: Could not load compiled mechanisms")
            
    except Exception as e:
        print(f"Error with mechanism loading: {e}")

    # Parameters
    initial_neuron_concentrations = [
        0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1
    ]
    
    neuron_parameters = [
        5.0e-3, 0.01, 1.0, 0.9, 5.0e-4, 0.2, 0.1, 1.0, 0.9, 0.005, 0.3, 0.1, 0.0001, 0.00001,
        0.0005, 0.0005, 0.0005, 0.0005, 0.9, 0.1, 0.1, 0.9, 0.0011, 0.0001, 0.0001, 0.00001,
        50.0, 1.0, 1.0, 0.0001, -65.0, -20.0
    ]

    # Create network
    grid_rows = 4
    grid_cols = 4
    network = CorrectedNeuronalNetwork(
        rows=grid_rows, 
        cols=grid_cols,
        initial_neuron_concentrations=initial_neuron_concentrations,
        neuron_params=neuron_parameters,
        synapse_type="ProbabilisticSyn",
        initial_syn_weight=0.6,    # Higher initial weight
        learning_rate=0.002,       # Higher learning rate
        min_weight=0.05,           # Prevent complete decay
        max_weight=1.0,
        threshold_growth=0.1,      # Lower thresholds for easier triggering
        threshold_apoptosis=-0.1
    )

    print(f"\n--- Network Summary ---")
    print(f"Dimensions: {network.rows}x{network.cols}")
    print(f"Total Neurons: {len([n for row in network.neurons for n in row])}")
    print(f"Total Connections: {len(network.connections)}")
    print("\n=== TESTING STIMULATION ===")
    # RIGHT AFTER creating the network, add this test:
    
    # Add varied stimulation to trigger different behaviors
    print("\nAdding stimulations...")
    network.neurons[0][0].add_external_current_stim(delay=50, dur=200, amp=1.0)   # Much stronger
    network.neurons[0][1].add_external_current_stim(delay=100, dur=200, amp=0.8)   
    network.neurons[1][0].add_external_current_stim(delay=150, dur=200, amp=0.9)   
    network.neurons[1][1].add_external_current_stim(delay=200, dur=200, amp=0.7)   # Increased
    # Setup NEURON simulation
    h.finitialize(-65)
    h.t = 0
    h.dt = 0.025

    # Simulation parameters
    runtime = 250  # ms
    print_interval = 3600  # Print every 200 time steps (5ms intervals)
    vis_interval = 100     # Visualize every 800 time steps (20ms intervals)
    num_steps = int(runtime / h.dt)

    print(f"\n--- Starting Simulation ({num_steps} steps, {runtime}ms total) ---")
    
    # Enable interactive plotting
    plt.ion()
    
    # Data collection for final plots
    time_data = []
    neuron_data = []
    weight_data = []
    
    # Initialize data collection
    for r in range(grid_rows):
        neuron_row = []
        for c in range(grid_cols):
            neuron_row.append({
                'voltage': [],
                'activity': [],
                'syn_input': [],
                'state': [],
                'P': [],
                'B': []
            })
        neuron_data.append(neuron_row)
    
    # Track first connection for weight plotting
    if len(network.connections) > 0:
        target_connection = network.connections[0]
        weight_data = []
    
    step = 0
    try:
        while h.t < runtime:
            h.fadvance()
            
            # Collect data every step
            time_data.append(h.t)
            
            for r in range(grid_rows):
                for c in range(grid_cols):
                    neuron = network.neurons[r][c]
                    try:
                        voltage = neuron.soma_segment.hoc.v
                        activity = neuron.ode_mech.activity_level
                        syn_input = neuron.ode_mech.syn_input_activity
                        state = neuron.calculate_and_get_neuron_state()
                        P = neuron.ode_mech.P
                        B = neuron.ode_mech.B
                        
                        neuron_data[r][c]['voltage'].append(voltage)
                        neuron_data[r][c]['activity'].append(activity)
                        neuron_data[r][c]['syn_input'].append(syn_input)
                        neuron_data[r][c]['state'].append(state)
                        neuron_data[r][c]['P'].append(P)
                        neuron_data[r][c]['B'].append(B)
                    except Exception as e:
                        print(f"Data collection error for {neuron.name}: {e}")
            
            # Collect weight data
            if len(network.connections) > 0:
                weight_data.append(target_connection['netcon_to_synapse'].weight[0])
            
            # Print network state at intervals
            if step % print_interval == 0:
                network.print_network_state(step, h.t)
            
            # Visualize network at intervals
            if step % vis_interval == 0:
                try:
                    network.visualize_network(step, h.t, save_fig=False)
                except Exception as e:
                    print(f"Visualization error at step {step}: {e}")
            
            # Apply synaptic weight modulation
            network.modulate_synaptic_weights()
            
            step += 1
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nSimulation error: {e}")
    
    print(f"\n--- Simulation Completed at t={h.t:.1f}ms ---")
    
    # Final state and visualization
    network.print_network_state(step, h.t)
    network.visualize_network(step, h.t, save_fig=False)
    
    # Create comprehensive final plots
    print("\nGenerating final analysis plots...")
    
    times = np.array(time_data)
    
    # Plot 1: All neuron voltages
    fig1, axes1 = plt.subplots(grid_rows, grid_cols, figsize=(15, 10), sharex=True, sharey=True)
    fig1.suptitle('Membrane Potentials of All Neurons', fontsize=16)
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            ax = axes1[r, c] if grid_rows > 1 else axes1[c]
            voltages = np.array(neuron_data[r][c]['voltage'])
            ax.plot(times, voltages, 'b-', linewidth=1)
            ax.axhline(-20, color='r', linestyle='--', alpha=0.5, label='Spike threshold')
            ax.set_title(f'Neuron ({r},{c})')
            ax.grid(True, alpha=0.3)
            if r == grid_rows-1:
                ax.set_xlabel('Time (ms)')
            if c == 0:
                ax.set_ylabel('Voltage (mV)')
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Plot 2: Synaptic activity and states
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Network Activity and States', fontsize=16)
    
    # Synaptic inputs
    axes2[0,0].set_title('Synaptic Input Activity')
    for r in range(grid_rows):
        for c in range(grid_cols):
            syn_inputs = np.array(neuron_data[r][c]['syn_input'])
            axes2[0,0].plot(times, syn_inputs, label=f'N({r},{c})')
    axes2[0,0].set_ylabel('Synaptic Input')
    axes2[0,0].legend()
    axes2[0,0].grid(True)
    
    # Activity levels
    axes2[0,1].set_title('Activity Levels (ODE States)')
    for r in range(grid_rows):
        for c in range(grid_cols):
            activities = np.array(neuron_data[r][c]['activity'])
            axes2[0,1].plot(times, activities, label=f'N({r},{c})')
    axes2[0,1].set_ylabel('Activity Level')
    axes2[0,1].legend()
    axes2[0,1].grid(True)
    
    # Neuron states
    axes2[1,0].set_title('Neuron States (Growth-Apoptosis)')
    for r in range(grid_rows):
        for c in range(grid_cols):
            states = np.array(neuron_data[r][c]['state'])
            axes2[1,0].plot(times, states, label=f'N({r},{c})')
    axes2[1,0].axhline(network.threshold_growth, color='g', linestyle=':', alpha=0.7, label='Growth threshold')
    axes2[1,0].axhline(network.threshold_apoptosis, color='r', linestyle=':', alpha=0.7, label='Apoptosis threshold')
    axes2[1,0].set_ylabel('State')
    axes2[1,0].set_xlabel('Time (ms)')
    axes2[1,0].legend()
    axes2[1,0].grid(True)
    
    # Synaptic weights
    if len(weight_data) > 0:
        axes2[1,1].set_title(f'Synaptic Weight Evolution\n({target_connection["pre_neuron"].name} → {target_connection["post_neuron"].name})')
        axes2[1,1].plot(times[:len(weight_data)], weight_data, 'purple', linewidth=2)
        axes2[1,1].set_ylabel('Synaptic Weight')
        axes2[1,1].set_xlabel('Time (ms)')
        axes2[1,1].grid(True)
    else:
        axes2[1,1].text(0.5, 0.5, 'No weight data available', ha='center', va='center', transform=axes2[1,1].transAxes)
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Plot 3: Neurotrophic factors
    fig3, axes3 = plt.subplots(grid_rows, grid_cols, figsize=(15, 10))
    fig3.suptitle('Neurotrophic Factors (proBDNF and BDNF)', fontsize=16)
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            ax = axes3[r, c] if grid_rows > 1 else axes3[c]
            P_data = np.array(neuron_data[r][c]['P'])
            B_data = np.array(neuron_data[r][c]['B'])
            ax.plot(times, P_data, 'r-', label='proBDNF', linewidth=2)
            ax.plot(times, B_data, 'g-', label='BDNF', linewidth=2)
            ax.set_title(f'Neuron ({r},{c})')
            ax.set_ylabel('Concentration (μM)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            if r == grid_rows-1:
                ax.set_xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.show()
    
    # Final summary
    print(f"\n=== Final Network Summary ===")
    for r in range(grid_rows):
        for c in range(grid_cols):
            neuron = network.neurons[r][c]
            final_state = neuron.calculate_and_get_neuron_state()
            final_P = neuron.ode_mech.P
            final_B = neuron.ode_mech.B
            final_activity = neuron.ode_mech.activity_level
            print(f"{neuron.name}: State={final_state:.3f}, P={final_P:.3f}μM, B={final_B:.3f}μM, Activity={final_activity:.3f}")
    
    if len(network.connections) > 0:
        print(f"\nFinal synaptic weights (sample):")
        for i, conn in enumerate(network.connections[:4]):
            weight = conn['netcon_to_synapse'].weight[0]
            print(f"  {conn['pre_neuron'].name} → {conn['post_neuron'].name}: {weight:.4f}")
    
    print(f"\nSimulation completed successfully!")
    
    # Keep plots open
    plt.ioff()
    input("Press Enter to close all plots...")
    plt.close('all')
    