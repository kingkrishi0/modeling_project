
from neuron import h, nrn
from neuronpp.cells.cell import Cell
from neuronpp.core.hocwrappers.synapses.single_synapse import SingleSynapse
import numpy as np
from neuronpp.core.hocwrappers.point_process import PointProcess
import os
from ode_neuron_class import ODENeuron
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MinimalBiologicalNetwork:
    def __init__(self, rows: int, cols: int, initial_neuron_concentrations: list, base_neuron_params: list,
                 synapse_type: str = "ProbabilisticSyn",
                 initial_syn_weight: float = 0.5,
                 learning_rate: float = 0.001,
                 min_weight: float = 0.01,
                 max_weight: float = 1.0,
                 prune_threshold: float = 0.35,
                 simulation_time: float = 20000.0):
        
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be above 0")
        
        self.rows = rows
        self.cols = cols
        self.num_neurons = rows * cols
        self.neurons = []
        self.connections = []
        self.stress_sources = []

        self._created_unidirectional_connections = set()
        self.synapse_type = synapse_type
        self.initial_syn_weight = initial_syn_weight
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.prune_threshold = prune_threshold
        self.simulation_time = simulation_time
        self._create_biologically_differentiated_neurons(initial_neuron_concentrations, base_neuron_params)
        self._build_grid_connections()
        self._initialize_activity_patterns(self.simulation_time)
        
    def _create_biologically_differentiated_neurons(self, initial_concentrations, base_params):        
        for r in range(self.rows):
            newrow = []
            for c in range(self.cols):
                layer_info = self._get_cortical_layer_properties(r)
                
               
                cell_size = self._calculate_cell_size(layer_info, r, c)
                
                activity_sensitivity = self._calculate_activity_sensitivity(layer_info, cell_size)
                
                neuron_params = self._adjust_bdnf_parameters_for_biology(
                    base_params, layer_info, cell_size, activity_sensitivity
                )
                neuron_concentrations = initial_concentrations.copy()
                
                neuron = BiologicalBDNFNeuron(
                    name=f'neuron_r{r}c{c}', 
                    initial_concentrations=neuron_concentrations, 
                    params=neuron_params,
                    grid_pos=(r, c),
                    cortical_layer=layer_info["name"],
                    cell_size=cell_size,
                    activity_sensitivity=activity_sensitivity
                )
                
                newrow.append(neuron)
            self.neurons.append(newrow)

    def _get_cortical_layer_properties(self, r):
        layer_map = {
            0: {"name": "Layer1", "cell_density": 0.2, "primary_type": "horizontal"},
            1: {"name": "Layer2/3", "cell_density": 1.2, "primary_type": "pyramidal_small"},
            2: {"name": "Layer4", "cell_density": 1.5, "primary_type": "granular"},
            3: {"name": "Layer5", "cell_density": 0.8, "primary_type": "pyramidal_large"},
            4: {"name": "Layer6", "cell_density": 1.0, "primary_type": "multiform"}
        }
        
        layer_index = min(r, 4)
        return layer_map[layer_index]

    def _calculate_cell_size(self, layer_info, r, c):
        base_sizes = {
            "horizontal": 0.3,      # Small horizontal cells
            "pyramidal_small": 0.7, # Small pyramidal neurons  
            "granular": 0.5,        # Medium granular cells
            "pyramidal_large": 1.8, # Large pyramidal neurons (real Layer 5)
            "multiform": 0.9        # Diverse cell types
        }
        
        base_size = base_sizes[layer_info["primary_type"]]
        
        position_variation = 1.0 + 0.3 * np.sin(c * np.pi / self.cols)
        
        return base_size * position_variation

    def _calculate_activity_sensitivity(self, layer_info, cell_size):
        
        layer_sensitivity = {
            "Layer1": 1.04,    # Horizontal cells - moderate sensitivity
            "Layer2/3": 1.1,  # Small pyramids - high sensitivity
            "Layer4": 1.11,    # Input layer - highest sensitivity
            "Layer5": 1.03,    # Large pyramids - moderate sensitivity
            "Layer6": 0.98     # Feedback layer - lower sensitivity
        }
        
        base_sensitivity = layer_sensitivity[layer_info["name"]]
        
        size_factor = 1.0 / (1.0 + cell_size * 0.2)
        
        return base_sensitivity * size_factor

    def _adjust_bdnf_parameters_for_biology(self, base_params, layer_info, cell_size, activity_sensitivity):
        """Adjust BDNF synthesis based on REAL biological properties"""
        
        params = base_params.copy()
        
        # Parameter indices from your ode_neuron.mod
        ksP_idx = 0          # proBDNF synthesis rate
        k_degB_idx = 9       # BDNF degradation
        ks_tPA_idx = 23      # tPA synthesis
        ks_p75_idx = 24      # p75 synthesis
        ks_TrkB_idx = 25     # TrkB synthesis
        activity_gain_idx = 27  # Activity gain parameter
        k_cleave_idx = 4  # Cleavage rate
        
        # 1. CELL SIZE EFFECTS (bigger cells need more BDNF but cost more ATP)
        if cell_size > 1.5:  # Large pyramidal neurons (Layer 5)
            # High baseline BDNF synthesis (they're metabolically active)
            params[ksP_idx] *= (1.0 + cell_size * 0.5)  # More proBDNF synthesis
            params[ks_TrkB_idx] *= (1.0 + cell_size * 0.3)  # More TrkB receptors
            # BUT higher degradation rates (metabolic cost)
            params[k_degB_idx] *= (1.0 + cell_size * 0.2)  # Faster BDNF turnover
            
        elif cell_size < 0.6:  # Small cells (Layer 1, granular)
            # Lower baseline synthesis but more efficient
            params[ksP_idx] *= (0.7 + cell_size * 0.3)  # Lower synthesis
            params[ks_tPA_idx] *= (1.0 + 0.2)  # More efficient processing
            params[k_degB_idx] *= (0.9)  # Slower degradation (efficient)
        
        # 2. LAYER-SPECIFIC BIOLOGY
        if layer_info["name"] == "Layer4":
            # Input layer - optimized for activity-dependent BDNF
            params[activity_gain_idx] *= activity_sensitivity
            params[ks_tPA_idx] *= 1.3  # Better BDNF processing
            #params[k_cleave_idx] *= 1.2  # More efficient cleavage
            
        elif layer_info["name"] == "Layer5":
            # Output layer - high baseline but vulnerable to activity loss
            params[ksP_idx] *= 1.2  # High baseline synthesis
            params[ks_p75_idx] *= 1.1  # Slightly more death receptors
            
        elif layer_info["name"] == "Layer2/3":
            # Associative layer - balanced but activity-sensitive
            params[activity_gain_idx] *= activity_sensitivity
            
        # 3. ACTIVITY SENSITIVITY (this is key for plasticity)
        params[activity_gain_idx] *= activity_sensitivity
        
        return params

    def _initialize_activity_patterns(self, simulation_time):
        """Initialize realistic activity patterns - periodic bursts with eventual quiet periods"""
        # Realistic activity parameters
        burst_duration = 50  # Short bursts (50ms)
        quiet_period_min = 200  # Minimum quiet time between bursts
        quiet_period_max = 800  # Maximum quiet time between bursts
        activity_stops_at = simulation_time * 0.8  # Activity stops at 70% of simulation time
        
        # Layer 4 gets most activity (sensory input)
        for c in range(self.cols):
            if self.rows > 2:
                layer4_neuron = self.neurons[2][c]
                stim_strength = 0.3 + 0.2 * np.sin(c * np.pi / self.cols)  # Reduced strength
                
                current_time = 20 + c * 10  # Staggered start
                
                while current_time < activity_stops_at:
                    # Add burst
                    layer4_neuron.add_external_current_stim(
                        delay=current_time, 
                        dur=burst_duration, 
                        amp=stim_strength
                    )
                    
                    # Random quiet period before next burst
                    quiet_time = np.random.randint(quiet_period_min, quiet_period_max)
                    current_time += burst_duration + quiet_time
        
        # Other layers get sporadic background activity
        for r in range(self.rows):
            for c in range(self.cols):
                if r != 2:  # Not Layer 4
                    neuron = self.neurons[r][c]
                    stim_strength = 0.05 + 0.05 * np.random.random()  # Much weaker
                    
                    current_time = 100 + np.random.randint(0, 500)
                    
                    while current_time < activity_stops_at:
                        # Sparse, random stimulation
                        if np.random.random() < 0.3:  # Only 30% chance of stimulation
                            neuron.add_external_current_stim(
                                delay=current_time,
                                dur=30,  # Very short
                                amp=stim_strength
                            )
                        
                        # Long random intervals
                        current_time += np.random.randint(800, 2000)
        
        print(f"📡 Activity patterns: bursts stop at {activity_stops_at:.0f}ms")
        print(f"🔇 Final {(1.0-0.7)*100:.0f}% of simulation will be quiet (testing BDNF decay)")
        
    def _get_neuron_at_grid_pos(self, r: int, c: int):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.neurons[r][c]
        return None
    
    def _create_connection(self, pre_neuron, post_neuron, pre_r, pre_c, post_r, post_c):
        """Create synaptic connection - weight determined by pre-neuron BDNF"""
        conn_id = f"{pre_neuron.name}_to_{post_neuron.name}"

        if conn_id in self._created_unidirectional_connections:
            return False
        
        # Simple distance constraint
        distance = np.sqrt((pre_r - post_r)**2 + (pre_c - post_c)**2)
        if distance > 2 or np.random.random() > np.exp(-distance / 2):
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
            
            # Initial weight based ONLY on pre-synaptic neuron's growth state
            # This is where BDNF controls connectivity!
            initial_weight = self.initial_syn_weight * (0.5 + 0.5 * pre_neuron.ode_mech.growth_strength)
            
            netcon_to_synapse = h.NetCon(pre_neuron.soma_segment.hoc._ref_v, 
                                       point_process_mech.hoc, 
                                       sec=pre_neuron.soma.hoc)
            netcon_to_synapse.weight[0] = initial_weight
            netcon_to_synapse.delay = 1.0
            netcon_to_synapse.threshold = -25.0

            try:
                h.setpointer(post_neuron.ode_mech._ref_syn_input_activity, 
                           'target_syn_input_activity', 
                           point_process_mech.hoc)
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
                'post_c': post_c,
                'is_pruned': False
            }
            
            self.connections.append(connection_info)
            self._created_unidirectional_connections.add(conn_id)
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to create connection {conn_id}: {e}")
            return False

    def _build_grid_connections(self):
        """Build connectivity based on distance"""
        if self.num_neurons < 2:
            print("Need at least 2 neurons")
            return

        connection_count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                pre_neuron = self._get_neuron_at_grid_pos(r, c)
                
                # Connect to nearby neurons
                for dr in range(-2, 3):  # 5x5 neighborhood
                    for dc in range(-2, 3):
                        if dr == 0 and dc == 0:
                            continue
                            
                        post_r, post_c = r + dr, c + dc
                        post_neuron = self._get_neuron_at_grid_pos(post_r, post_c)

                        if post_neuron:
                            if self._create_connection(pre_neuron, post_neuron, r, c, post_r, post_c):
                                connection_count += 1
        
        print(f"Total connections created: {connection_count}")

    def bdnf_driven_plasticity(self):
        """Synaptic plasticity driven PURELY by BDNF/proBDNF levels"""
        pruned_connections = []
        
        for i, conn_info in enumerate(self.connections):
            if conn_info['is_pruned']:
                continue
                
            pre_neuron = conn_info['pre_neuron']
            post_neuron = conn_info['post_neuron']
            netcon_to_synapse = conn_info['netcon_to_synapse']

            # Get BDNF-related states from your ode_neuron.mod
            pre_growth = pre_neuron.ode_mech.growth_strength  # BDNF/TrkB signaling
            pre_apop = pre_neuron.ode_mech.apop_strength      # proBDNF/p75 signaling
            post_growth = post_neuron.ode_mech.growth_strength
            post_apop = post_neuron.ode_mech.apop_strength
            pre_signal = pre_neuron.calculate_and_get_neuron_state()
            post_signal = post_neuron.calculate_and_get_neuron_state()
            
            current_weight = netcon_to_synapse.weight[0]
            
            # Pure BDNF-based plasticity rule
            # High growth_strength = high BDNF = strengthen synapses
            # High apop_strength = high proBDNF = weaken synapses
            
            if pre_signal >= -0.14 and post_signal >= -0.14:
                # Both neurons have high BDNF - strengthen synapse (LTP-like)
                weight_change = self.learning_rate * max(pre_growth, post_growth)
            elif pre_signal < -0.23 and post_signal < -0.23:
                # High proBDNF/p75 signaling - weaken synapse (LTD-like)
                weight_change = -self.learning_rate * max(pre_apop, post_apop)
            else:
                # Maintenance level - slight decay without BDNF support
                weight_change = -self.learning_rate * 0.149
            
            new_weight = current_weight + weight_change
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            
            # Synaptic pruning when BDNF support is insufficient
            if new_weight < self.prune_threshold:
                conn_info['is_pruned'] = True
                netcon_to_synapse.weight[0] = 0.0
                pruned_connections.append(i)
            else:
                netcon_to_synapse.weight[0] = new_weight

        return len(pruned_connections)

    def get_network_health_metrics(self):
        """Calculate network health metrics"""
        total_neurons = self.rows * self.cols
        active_connections = sum(1 for conn in self.connections if not conn['is_pruned'])
        
        avg_growth = np.mean([n.ode_mech.growth_strength for row in self.neurons for n in row])
        avg_apop = np.mean([n.ode_mech.apop_strength for row in self.neurons for n in row])
        
        # Calculate average BDNF and proBDNF levels
        avg_bdnf = np.mean([n.ode_mech.B for row in self.neurons for n in row])
        avg_probdnf = np.mean([n.ode_mech.P for row in self.neurons for n in row])
        
        return {
            'active_connections': active_connections,
            'connection_density': active_connections / (total_neurons * (total_neurons - 1)),
            'avg_growth_signal': avg_growth,
            'avg_apoptosis_signal': avg_apop,
            'avg_bdnf': avg_bdnf,
            'avg_probdnf': avg_probdnf,
            'network_health': avg_growth - avg_apop
        }

    def visualize_bdnf_network(self, time_step, current_time, save_fig=False):
        """Visualize network with BDNF states"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        # Calculate neuron positions
        neuron_positions = {}
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * 3.0
                y = (self.rows - r - 1) * 3.0
                neuron_positions[(r, c)] = (x, y)
        
        # Draw connections colored by weight (BDNF-determined)
        for conn in self.connections:
            if conn['is_pruned']:
                continue
                
            if all(k in conn for k in ['pre_r', 'pre_c', 'post_r', 'post_c']):
                pre_pos = neuron_positions[(conn['pre_r'], conn['pre_c'])]
                post_pos = neuron_positions[(conn['post_r'], conn['post_c'])]
                
                weight = conn['netcon_to_synapse'].weight[0]
                thickness = max(0.3, min(6.0, weight * 8))
                
                # Color based on BDNF-driven weight
                if weight > 0.5:
                    color = 'darkgreen'  # High BDNF = strong synapse
                elif weight > 0.3:
                    color = 'orange'     # Medium BDNF
                else:
                    color = 'red'        # Low BDNF = weak synapse
                
                ax.plot([pre_pos[0], post_pos[0]], [pre_pos[1], post_pos[1]], 
                       color=color, linewidth=thickness, alpha=0.6)
        
        # Draw neurons colored by BDNF/proBDNF balance
        for r in range(self.rows):
            for c in range(self.cols):
                neuron = self.neurons[r][c]
                pos = neuron_positions[(r, c)]
                
                # Color based on BDNF vs proBDNF
                bdnf = neuron.ode_mech.B
                probdnf = neuron.ode_mech.P
                growth = neuron.ode_mech.growth_strength
                apop = neuron.ode_mech.apop_strength
                
                net_health = growth - apop
                
                if net_health > 0.2:     # ✅ Higher threshold
                    neuron_color = 'lightgreen'
                elif net_health > -0.1:   # ✅ Wider range
                    neuron_color = 'yellow'        # Balanced
                elif net_health > -0.2:
                    neuron_color = 'orange'        # High proBDNF
                else:
                    neuron_color = 'red'           # Very high proBDNF
                
                # Size based on cell size and activity
                activity = neuron.ode_mech.activity_level
                base_size = 400 + neuron.cell_size * 300
                size = max(200, min(1500, base_size + activity * 100))
                
                ax.scatter(pos[0], pos[1], s=size, c=neuron_color, 
                          edgecolors='black', linewidth=2, alpha=0.8)
                
                # Label with BDNF info
                ax.text(pos[0], pos[1], f'({r},{c})\nB:{bdnf:.2f}\nP:{probdnf:.2f}\nH:{net_health:.2f}',
                       ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Mark cortical layers
        for r in range(self.rows):
            layer_info = self._get_cortical_layer_properties(r)
            ax.text(-2, (self.rows - r - 1) * 3, layer_info["name"], rotation=90, 
                   ha='center', va='center', fontsize=10, alpha=0.7)
        
        ax.set_xlim(-3, self.cols * 3)
        ax.set_ylim(-1, self.rows * 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'BDNF-Driven Neural Network - t={current_time:.1f}ms\nActivity-Dependent BDNF Synthesis', 
                    fontsize=14)
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        
        # Legend
        legend_elements = [
            plt.scatter([], [], s=300, c='lightgreen', edgecolors='black', label='High BDNF'),
            plt.scatter([], [], s=300, c='yellow', edgecolors='black', label='Balanced'),
            plt.scatter([], [], s=300, c='orange', edgecolors='black', label='High proBDNF'),
            plt.scatter([], [], s=300, c='red', edgecolors='black', label='Very High proBDNF'),
            plt.Line2D([0], [0], color='darkgreen', lw=4, label='Strong Synapse (High BDNF)'),
            plt.Line2D([0], [0], color='orange', lw=4, label='Medium Synapse'),
            plt.Line2D([0], [0], color='red', lw=4, label='Weak Synapse (Low BDNF)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'bdnf_network_t{time_step:04d}.png', dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.1)


class BiologicalBDNFNeuron(ODENeuron):
    """Extended ODENeuron with biological properties"""
    
    def __init__(self, name, initial_concentrations, params, grid_pos=(0,0), 
                 cortical_layer="Layer4", cell_size=1.0, activity_sensitivity=1.0):
        super().__init__(name, initial_concentrations, params)
        
        self.grid_pos = grid_pos
        self.cortical_layer = cortical_layer
        self.cell_size = cell_size
        self.activity_sensitivity = activity_sensitivity
        
        print(f"BiologicalBDNFNeuron '{self.name}' - Layer: {cortical_layer}, "
              f"Size: {cell_size:.2f}, Sensitivity: {activity_sensitivity:.2f}")
    
    def calculate_and_get_neuron_state(self) -> float:
        """Calculate state based purely on BDNF balance"""
        growth_strength = self.ode_mech.growth_strength
        apop_strength = self.ode_mech.apop_strength
        signal = growth_strength - apop_strength
        self.neuron_state = signal
        return self.neuron_state


def test_ratio_tipping_point(target_ratio, base_k_degB=0.0145, base_ksP=2.0e-3, base_k_cleave=0.008):
    """Test a specific BDNF:proBDNF ratio and save results with network visualizations"""
    
    # Calculate parameters for target ratio using the formula
    k_degB = base_k_degB * (target_ratio / 4.0) ** 0.7
    ksP = base_ksP * (target_ratio / 4.0) ** 0.5
    k_cleave = base_k_cleave * (4.0 / target_ratio) ** 0.6
    
    print(f"\n🧪 Testing Ratio 1:{target_ratio}")
    print(f"   k_degB: {k_degB:.5f}, ksP: {ksP:.5f}, k_cleave: {k_cleave:.5f}")
    
    # Modify base parameters - CORRECT INDICES
    test_params = base_neuron_parameters.copy()
    test_params[3] = ksP      # ksP index (CORRECT)
    test_params[4] = k_cleave # k_cleave index (CORRECT)
    test_params[12] = k_degB  # k_degB index (CORRECT)
    
    # Create network with modified parameters
    network = MinimalBiologicalNetwork(
        rows=4, cols=10,
        initial_neuron_concentrations=initial_neuron_concentrations,
        base_neuron_params=test_params,
        synapse_type="ProbabilisticSyn",
        initial_syn_weight=0.5,
        learning_rate=0.005,
        min_weight=0.01,
        max_weight=1.0,
        prune_threshold=0.05,
        simulation_time=30000.0  # Use parameter instead of global
    )
    
    # Create results directory
    ratio_dir = f"ratio_1_{target_ratio:.1f}_results"
    os.makedirs(ratio_dir, exist_ok=True)
    
    # Define visualization time points
    viz_times = {
        'initial': 1000.0,    # Initial state (1 second)
        'middle': 20000.0,    # Middle state (20 seconds)
        'final': 29900.0      # Near final state (29.9 seconds)
    }
    
    # Run simulation
    h.finitialize(-65)
    current_time = 0.0
    time_step = 0
    last_plasticity_update = 0.0
    
    time_points = []
    health_metrics = []
    initial_connections = len([c for c in network.connections if not c['is_pruned']])
    
    print(f"📊 Starting simulation with {initial_connections} initial connections...")
    print(f"📸 Will capture network visualizations at: {list(viz_times.values())} ms")
    
    while current_time <= 30000.0:  # Fixed simulation time
        h.fadvance()
        current_time = h.t
        time_step += 1
        
        # Update plasticity every 10ms
        if current_time - last_plasticity_update >= 10.0:
            network.bdnf_driven_plasticity()
            last_plasticity_update = current_time
        
        # Check if we should capture a network visualization
        for viz_name, viz_time in viz_times.items():
            if abs(current_time - viz_time) < 5.0:  # Within 5ms of target time
                print(f"📸 Capturing {viz_name} network state at t={current_time:.1f}ms")
                
                # Create custom visualization using existing method
                fig, ax = plt.subplots(1, 1, figsize=(14, 12))
                
                # Calculate neuron positions
                neuron_positions = {}
                for r in range(network.rows):
                    for c in range(network.cols):
                        x = c * 3.0
                        y = (network.rows - r - 1) * 3.0
                        neuron_positions[(r, c)] = (x, y)
                
                # Draw connections colored by weight (BDNF-determined)
                for conn in network.connections:
                    if conn['is_pruned']:
                        continue
                        
                    if all(k in conn for k in ['pre_r', 'pre_c', 'post_r', 'post_c']):
                        pre_pos = neuron_positions[(conn['pre_r'], conn['pre_c'])]
                        post_pos = neuron_positions[(conn['post_r'], conn['post_c'])]
                        
                        weight = conn['netcon_to_synapse'].weight[0]
                        thickness = max(0.3, min(6.0, weight * 8))
                        
                        # Color based on BDNF-driven weight
                        if weight > 0.5:
                            color = 'darkgreen'  # High BDNF = strong synapse
                        elif weight > 0.3:
                            color = 'orange'     # Medium BDNF
                        else:
                            color = 'red'        # Low BDNF = weak synapse
                        
                        ax.plot([pre_pos[0], post_pos[0]], [pre_pos[1], post_pos[1]], 
                               color=color, linewidth=thickness, alpha=0.6)
                
                # Draw neurons colored by BDNF/proBDNF balance
                for r in range(network.rows):
                    for c in range(network.cols):
                        neuron = network.neurons[r][c]
                        pos = neuron_positions[(r, c)]
                        
                        # Color based on BDNF vs proBDNF
                        bdnf = neuron.ode_mech.B
                        probdnf = neuron.ode_mech.P
                        growth = neuron.ode_mech.growth_strength
                        apop = neuron.ode_mech.apop_strength
                        
                        net_health = growth - apop
                        
                        if net_health > 0.2:
                            neuron_color = 'lightgreen'
                        elif net_health > -0.1:
                            neuron_color = 'yellow'
                        elif net_health > -0.2:
                            neuron_color = 'orange'
                        else:
                            neuron_color = 'red'
                        
                        # Size based on cell size and activity
                        activity = neuron.ode_mech.activity_level
                        base_size = 400 + neuron.cell_size * 300
                        size = max(200, min(1500, base_size + activity * 100))
                        
                        ax.scatter(pos[0], pos[1], s=size, c=neuron_color, 
                                  edgecolors='black', linewidth=2, alpha=0.8)
                        
                        # Label with BDNF info
                        ax.text(pos[0], pos[1], f'({r},{c})\nB:{bdnf:.2f}\nP:{probdnf:.2f}\nH:{net_health:.2f}',
                               ha='center', va='center', fontsize=8, fontweight='bold')
                
                # Mark cortical layers
                for r in range(network.rows):
                    layer_info = network._get_cortical_layer_properties(r)
                    ax.text(-2, (network.rows - r - 1) * 3, layer_info["name"], rotation=90, 
                           ha='center', va='center', fontsize=10, alpha=0.7)
                
                ax.set_xlim(-3, network.cols * 3)
                ax.set_ylim(-1, network.rows * 3)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_title(f'BDNF Network - Ratio 1:{target_ratio} - {viz_name.title()} State - t={current_time:.1f}ms', 
                            fontsize=14)
                ax.set_xlabel('Grid X')
                ax.set_ylabel('Grid Y')
                
                # Legend
                legend_elements = [
                    plt.scatter([], [], s=300, c='lightgreen', edgecolors='black', label='High BDNF'),
                    plt.scatter([], [], s=300, c='yellow', edgecolors='black', label='Balanced'),
                    plt.scatter([], [], s=300, c='orange', edgecolors='black', label='High proBDNF'),
                    plt.scatter([], [], s=300, c='red', edgecolors='black', label='Very High proBDNF'),
                    plt.Line2D([0], [0], color='darkgreen', lw=4, label='Strong Synapse (High BDNF)'),
                    plt.Line2D([0], [0], color='orange', lw=4, label='Medium Synapse'),
                    plt.Line2D([0], [0], color='red', lw=4, label='Weak Synapse (Low BDNF)')
                ]
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
                
                plt.tight_layout()
                
                # Save the network visualization
                filename = f"{ratio_dir}/network_{viz_name}_t{current_time:.0f}ms.png"
                # Ensure directory exists before saving
                os.makedirs(ratio_dir, exist_ok=True)
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Saved network visualization: {filename}")
                
                # Remove this time point so we don't capture it again
                del viz_times[viz_name]
                break
        
        # Collect metrics every 100ms
        if int(current_time) % 100 == 0:
            metrics = network.get_network_health_metrics()
            time_points.append(current_time)
            health_metrics.append(metrics)
    
    # Calculate key metrics
    connection_counts = [m['active_connections'] for m in health_metrics]
    final_connections = connection_counts[-1]
    connection_survival = (final_connections / initial_connections) * 100
    
    # Find 50% connection loss time
    half_connections = initial_connections * 0.5
    time_50_loss = "No 50% loss"
    for i, count in enumerate(connection_counts):
        if count <= half_connections:
            time_50_loss = f"{time_points[i]:.0f}ms"
            break
    
    # Find catastrophic failure time (10% remaining)
    catastrophic_threshold = initial_connections * 0.1
    time_catastrophic = "No catastrophic failure"
    for i, count in enumerate(connection_counts):
        if count <= catastrophic_threshold:
            time_catastrophic = f"{time_points[i]:.0f}ms"
            break
    
    # Calculate recovery potential (connections gained in last 5000ms)
    activity_stop_time = 30000.0 * 0.8  # 24000ms
    recovery_start_idx = None
    for i, t in enumerate(time_points):
        if t >= activity_stop_time:
            recovery_start_idx = i
            break
    
    if recovery_start_idx and recovery_start_idx < len(connection_counts) - 10:
        recovery_potential = connection_counts[-1] - connection_counts[recovery_start_idx]
    else:
        recovery_potential = 0
    
    # Save results text file
    with open(f"{ratio_dir}/analysis_summary.txt", 'w') as f:
        f.write(f"BDNF:proBDNF Ratio Analysis - 1:{target_ratio}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Parameters Used:\n")
        f.write(f"  k_degB: {k_degB:.6f}\n")
        f.write(f"  ksP: {ksP:.6f}\n")
        f.write(f"  k_cleave: {k_cleave:.6f}\n\n")
        f.write(f"Key Metrics:\n")
        f.write(f"  Initial Connections: {initial_connections}\n")
        f.write(f"  Final Connections: {final_connections}\n")
        f.write(f"  Connection Survival: {connection_survival:.1f}%\n")
        f.write(f"  50% Connection Loss Time: {time_50_loss}\n")
        f.write(f"  Catastrophic Failure Time: {time_catastrophic}\n")
        f.write(f"  Recovery Potential: {recovery_potential} connections\n\n")
        if health_metrics:
            final_bdnf = health_metrics[-1]['avg_bdnf']
            final_probdnf = health_metrics[-1]['avg_probdnf']
            if final_bdnf > 0:
                f.write(f"Final proBDNF:BDNF Ratio: {final_probdnf/final_bdnf:.2f}\n\n")
        
        # Add network visualization info
        f.write(f"Network Visualizations Saved:\n")
        f.write(f"  • Initial state: network_initial_t1000ms.png\n")
        f.write(f"  • Middle state: network_middle_t20000ms.png\n")
        f.write(f"  • Final state: network_final_t29900ms.png\n")
    
    # Generate and save analysis plots
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Connection survival
    plt.subplot(2, 3, 1)
    times = np.array(time_points)
    plt.plot(times, connection_counts, 'purple', linewidth=2)
    plt.axhline(y=half_connections, color='orange', linestyle='--', label='50% Loss')
    plt.axhline(y=catastrophic_threshold, color='red', linestyle='--', label='10% Loss')
    
    # Mark visualization time points
    plt.axvline(x=1000, color='green', linestyle=':', alpha=0.7, label='Initial viz')
    plt.axvline(x=20000, color='blue', linestyle=':', alpha=0.7, label='Middle viz')
    plt.axvline(x=29900, color='red', linestyle=':', alpha=0.7, label='Final viz')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Active Connections')
    plt.title(f'Connection Survival - Ratio 1:{target_ratio}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: BDNF vs proBDNF
    plt.subplot(2, 3, 2)
    bdnf_values = [m['avg_bdnf'] for m in health_metrics]
    probdnf_values = [m['avg_probdnf'] for m in health_metrics]
    plt.plot(times, bdnf_values, 'g-', linewidth=2, label='BDNF')
    plt.plot(times, probdnf_values, 'r-', linewidth=2, label='proBDNF')
    plt.axvline(x=1000, color='green', linestyle=':', alpha=0.5)
    plt.axvline(x=20000, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(x=29900, color='red', linestyle=':', alpha=0.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Concentration (μM)')
    plt.title('BDNF vs proBDNF Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Network health
    plt.subplot(2, 3, 3)
    health_values = [m['network_health'] for m in health_metrics]
    plt.plot(times, health_values, 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=1000, color='green', linestyle=':', alpha=0.5)
    plt.axvline(x=20000, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(x=29900, color='red', linestyle=':', alpha=0.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Network Health')
    plt.title('Network Health Evolution')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Growth vs Apoptosis
    plt.subplot(2, 3, 4)
    growth_values = [m['avg_growth_signal'] for m in health_metrics]
    apop_values = [m['avg_apoptosis_signal'] for m in health_metrics]
    plt.plot(times, growth_values, 'g-', linewidth=2, label='Growth')
    plt.plot(times, apop_values, 'r-', linewidth=2, label='Apoptosis')
    plt.axvline(x=1000, color='green', linestyle=':', alpha=0.5)
    plt.axvline(x=20000, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(x=29900, color='red', linestyle=':', alpha=0.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal Strength')
    plt.title('Growth vs Apoptosis Signals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Ratio evolution
    plt.subplot(2, 3, 5)
    ratio_values = [m['avg_probdnf']/max(m['avg_bdnf'], 0.001) for m in health_metrics]
    plt.plot(times, ratio_values, 'orange', linewidth=2)
    plt.axhline(y=target_ratio, color='black', linestyle='--', label=f'Target 1:{target_ratio}')
    plt.axvline(x=1000, color='green', linestyle=':', alpha=0.5)
    plt.axvline(x=20000, color='blue', linestyle=':', alpha=0.5)
    plt.axvline(x=29900, color='red', linestyle=':', alpha=0.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('proBDNF:BDNF Ratio')
    plt.title('Actual Ratio Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary text plot
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.9, f"Ratio: 1:{target_ratio}", fontsize=14, fontweight='bold')
    plt.text(0.1, 0.8, f"Survival: {connection_survival:.1f}%", fontsize=12)
    plt.text(0.1, 0.7, f"50% Loss: {time_50_loss}", fontsize=12)
    plt.text(0.1, 0.6, f"Catastrophic: {time_catastrophic}", fontsize=12)
    plt.text(0.1, 0.5, f"Recovery: {recovery_potential}", fontsize=12)
    if health_metrics and health_metrics[-1]['avg_bdnf'] > 0:
        final_ratio = health_metrics[-1]['avg_probdnf']/health_metrics[-1]['avg_bdnf']
        plt.text(0.1, 0.4, f"Final Ratio: {final_ratio:.1f}", fontsize=12)
    
    plt.text(0.1, 0.2, "Network visualizations:", fontsize=10, style='italic')
    plt.text(0.1, 0.1, "Initial • Middle • Final", fontsize=10, style='italic')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Summary')
    
    plt.suptitle(f'Tipping Point Analysis - BDNF:proBDNF Ratio 1:{target_ratio}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{ratio_dir}/analysis_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    import pickle
    analysis_data = {
        'target_ratio': target_ratio,
        'parameters': {'k_degB': k_degB, 'ksP': ksP, 'k_cleave': k_cleave},
        'time_points': time_points,
        'health_metrics': health_metrics,
        'connection_survival': connection_survival,
        'time_50_loss': time_50_loss,
        'time_catastrophic': time_catastrophic,
        'recovery_potential': recovery_potential,
        'visualization_times': {
            'initial': 1000.0,
            'middle': 20000.0, 
            'final': 29900.0
        }
    }
    
    with open(f"{ratio_dir}/analysis_data.pkl", 'wb') as f:
        pickle.dump(analysis_data, f)
    
    print(f"✅ Results saved to {ratio_dir}/")
    print(f"📊 Final metrics: {connection_survival:.1f}% survival, 50% loss at {time_50_loss}")
    print(f"🖼️  Network visualizations saved: initial, middle, and final states")
    
    return connection_survival, time_50_loss, time_catastrophic, recovery_potential


# CORRECTED Main execution block
if __name__ == "__main__":
    print("=== Minimal Biological BDNF Neural Network ===")
    
    # Load your existing mechanisms (keep this part unchanged)
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
    
    # Use your existing parameters
    initial_neuron_concentrations = [0.2, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    base_neuron_parameters = [
        0.0001, #g_leak
        -65.0, #e_leak
        -20.0, #v_threshold_spike
        2.0e-3, # ksP
        0.008, # k_cleave
        1.0, # k_p75_pro_on
        0.9, # k_p75_pro_off
        5.0e-4, # k_degP
        0.2, # k_TrkB_pro_on
        0.1, # k_TrkB_pro_off
        1.0, # k_TrkB_B_on
        0.9, # k_TrkB_B_off
        0.0145, # k_degB
        0.3, # k_p75_B_on
        0.1, # k_p75_B_off
        0.0001, # k_degR1
        0.00001, # k_degR2
        0.0005, # k_int_p75_pro
        0.0005, # k_int_p75_B
        0.0005, # k_int_TrkB_B
        0.0005, # k_int_TrkB_pro
        0.9, # aff_p75_pro
        0.1, # aff_p75_B
        0.1, # aff_TrkB_pro
        0.9, # aff_TrkB_B
        0.0013, # k_deg_tPA
        0.001, # ks_tPA
        0.001, # ks_p75
        0.0001, # ks_TrkB
        50.0, # tau_activity
        0.001 # activity_gain
    ]
    
    print("=== BDNF:proBDNF Tipping Point Analysis ===")
    print("🎯 Fine-grained analysis between ratios 1:6 and 1:8 to find precise tipping point")
    
    # Test ratios for precise tipping point discovery between 1:6 and 1:8
    target_ratios = [7.45, 7.46, 7.47, 7.48, 7.49, 7.50,]
    
    # Summary results
    summary_results = []
    
    for ratio in target_ratios:
        print(f"\n🔬 Testing ratio 1:{ratio}...")
        survival, loss50, catastrophic, recovery = test_ratio_tipping_point(
            ratio, 
            base_neuron_parameters[12],  # k_degB
            base_neuron_parameters[3],   # ksP  
            base_neuron_parameters[4]    # k_cleave
        )
        summary_results.append({
            'ratio': ratio,
            'survival': survival,
            'loss50': loss50,
            'catastrophic': catastrophic,
            'recovery': recovery
        })
    
    # Create summary comparison
    with open("tipping_point_summary.txt", 'w') as f:
        f.write("BDNF:proBDNF Tipping Point Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Ratio':<8} {'Survival%':<12} {'50% Loss':<15} {'Catastrophic':<15} {'Recovery':<10}\n")
        f.write("-" * 60 + "\n")
        
        for result in summary_results:
            f.write(f"1:{result['ratio']:<6} {result['survival']:<12.1f} "
                   f"{result['loss50']:<15} {result['catastrophic']:<15} {result['recovery']:<10}\n")
    
    print(f"\n✅ Tipping point analysis complete!")
    print(f"📁 Check individual ratio folders and tipping_point_summary.txt")
    print(f"🎯 Look for ratios where survival drops below 50% to identify tipping point")
    print(f"🖼️  Each ratio folder contains:")
    print(f"   • analysis_plots.png (time series analysis)")
    print(f"   • network_initial_t1000ms.png (network at 1 second)")
    print(f"   • network_middle_t20000ms.png (network at 20 seconds)")
    print(f"   • network_final_t29900ms.png (network near end)")
    print(f"   • analysis_summary.txt (detailed metrics)")
    print(f"   • analysis_data.pkl (raw data for further analysis)")