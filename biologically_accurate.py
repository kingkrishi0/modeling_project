"""
Biologically Accurate Neural Network with Localized Differentiation
Implements neurodegeneration cascades, synaptic pruning, and regional vulnerability
"""

from neuron import h, nrn
from neuronpp.cells.cell import Cell
from neuronpp.core.hocwrappers.synapses.single_synapse import SingleSynapse
import numpy as np
from neuronpp.core.hocwrappers.point_process import PointProcess
import os
from ode_neuron_class import ODENeuron
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BiologicalNeuralNetwork:
    def __init__(self, rows: int, cols: int, initial_neuron_concentrations: list, base_neuron_params: list,
                 synapse_type: str = "ProbabilisticSyn",
                 initial_syn_weight: float = 0.5,
                 learning_rate: float = 0.001,
                 min_weight: float = 0.01,
                 max_weight: float = 1.0,
                 prune_threshold: float = 0.05):  # Threshold for synaptic pruning
        
        if rows<=0 or cols<=0:
            raise ValueError("rows and cols must be above 0")
        
        self.rows = rows
        self.cols = cols
        self.num_neurons = rows * cols
        self.neurons = []
        self.connections = []
        self.neuron_vulnerability = {}  # Track regional vulnerability
        self.stress_sources = []  # Locations of pathological stress

        self._created_unidirectional_connections = set()
        self.synapse_type = synapse_type
        self.initial_syn_weight = initial_syn_weight
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.prune_threshold = prune_threshold

        print(f"Building biologically accurate neural network ({rows}x{cols})")

        # Create neurons with spatial heterogeneity
        self._create_heterogeneous_neurons(initial_neuron_concentrations, base_neuron_params)
        self._build_grid_connections()
        self._initialize_pathological_stress()
        
        print(f"Network created with {len(self.connections)} connections!")
        print(f"Stress sources at: {self.stress_sources}")

    def _create_heterogeneous_neurons(self, initial_concentrations, base_params):
        """Create neurons with biologically accurate spatial differentiation"""
        
        for r in range(self.rows):
            newrow = []
            for c in range(self.cols):
                # Calculate spatial factors
                center_r, center_c = self.rows / 2, self.cols / 2
                distance_from_center = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                
                # Create regional vulnerability patterns (like cortical layers)
                cortical_layer = self._get_cortical_layer(r, c)
                vulnerability = self._calculate_vulnerability(r, c, cortical_layer)
                
                # Modify parameters based on location and vulnerability
                neuron_params = self._customize_neuron_parameters(base_params, r, c, vulnerability)
                neuron_concentrations = self._customize_initial_concentrations(initial_concentrations, vulnerability)
                
                neuron = BiologicalODENeuron(
                    name=f'neuron_r{r}c{c}', 
                    initial_concentrations=neuron_concentrations, 
                    params=neuron_params,
                    grid_pos=(r, c),
                    vulnerability=vulnerability,
                    cortical_layer=cortical_layer
                )
                
                # Store vulnerability info
                self.neuron_vulnerability[(r, c)] = vulnerability
                
                newrow.append(neuron)
            self.neurons.append(newrow)

    def _get_cortical_layer(self, r, c):
        """Simulate cortical layer organization"""
        # Different layers have different properties
        layer_map = {
            0: "Layer1",      # Molecular layer - few cell bodies
            1: "Layer2/3",    # Superficial pyramidal - vulnerable to stress
            2: "Layer4",      # Granular - input layer
            3: "Layer5",      # Deep pyramidal - projection neurons
            4: "Layer6"       # Multiform - diverse cell types
        }
        layer_index = min(r, 4)  # Map rows to cortical layers
        return layer_map[layer_index]

    def _calculate_vulnerability(self, r, c, cortical_layer):
        """Calculate neuronal vulnerability based on biological factors"""
        base_vulnerability = 0.5
        
        # Layer-specific vulnerability (based on neurodegeneration research)
        layer_vulnerability = {
            "Layer1": 0.3,      # Fewer neurons, less vulnerable
            "Layer2/3": 0.8,    # High metabolic demand, very vulnerable
            "Layer4": 0.5,      # Moderate vulnerability
            "Layer5": 0.7,      # Large neurons, high energy needs
            "Layer6": 0.4       # More resilient
        }
        
        # Distance from vasculature (center = good blood supply)
        center_r, center_c = self.rows / 2, self.cols / 2
        distance_factor = np.sqrt((r - center_r)**2 + (c - center_c)**2) / (max(self.rows, self.cols) / 2)
        
        # Age-related vulnerability gradient
        age_factor = 0.1 + 0.3 * (r / self.rows)  # Older neurons more vulnerable
        
        vulnerability = layer_vulnerability[cortical_layer] + 0.2 * distance_factor + age_factor
        return min(1.0, max(0.1, vulnerability))

    def _customize_neuron_parameters(self, base_params, r, c, vulnerability):
        """Customize neuron parameters based on location and vulnerability"""
        params = base_params.copy()
        
        # Indices for key parameters
        ksP_idx = 0          # proBDNF synthesis
        k_degP_idx = 4       # proBDNF degradation  
        k_degB_idx = 9       # BDNF degradation
        ks_tPA_idx = 23      # tPA synthesis
        k_deg_tPA_idx = 22   # tPA degradation
        ks_p75_idx = 24      # p75 synthesis
        ks_TrkB_idx = 25     # TrkB synthesis
        activity_gain_idx = 27  # Activity sensitivity
        
        # Vulnerable neurons: higher stress protein production, lower neuroprotection
        if vulnerability > 0.7:  # High vulnerability
            params[ksP_idx] *= (1.5 + 0.3 * vulnerability)     # More stress-induced proBDNF
            params[k_degB_idx] *= (1.2 + 0.2 * vulnerability)  # Faster BDNF degradation
            params[ks_p75_idx] *= (1.3 + 0.4 * vulnerability)  # More death receptors
            params[ks_TrkB_idx] *= (0.7 - 0.2 * vulnerability) # Fewer survival receptors
            params[activity_gain_idx] *= (1.2 + 0.3 * vulnerability)  # More activity-sensitive
            
        elif vulnerability < 0.4:  # Low vulnerability (resilient neurons)
            params[ksP_idx] *= (0.8 - 0.1 * vulnerability)     # Less stress response
            params[k_degB_idx] *= (0.8 - 0.1 * vulnerability)  # Slower BDNF degradation
            params[ks_TrkB_idx] *= (1.2 + 0.3 * vulnerability) # More survival receptors
            params[ks_tPA_idx] *= (1.1 + 0.2 * vulnerability)  # Better BDNF processing
        
        # Regional gradients (simulate developmental/genetic differences)
        regional_factor = 1.0 + 0.2 * np.sin(r * np.pi / self.rows) * np.cos(c * np.pi / self.cols)
        params[ksP_idx] *= regional_factor
        params[ks_TrkB_idx] *= (2.0 - regional_factor)  # Inverse relationship
        
        return params

    def _customize_initial_concentrations(self, base_concentrations, vulnerability):
        """Adjust initial concentrations based on vulnerability"""
        concentrations = base_concentrations.copy()
        
        # Higher vulnerability = more baseline stress
        if vulnerability > 0.6:
            concentrations[0] *= (1.0 + 0.5 * vulnerability)  # More proBDNF
            concentrations[1] *= (0.8 - 0.2 * vulnerability)  # Less BDNF
            concentrations[2] *= (0.9 - 0.1 * vulnerability)  # Fewer p75 receptors initially
            concentrations[3] *= (0.8 - 0.3 * vulnerability)  # Fewer TrkB receptors
        
        return concentrations

    def _initialize_pathological_stress(self):
        """Initialize pathological stress sources (like amyloid plaques, tau tangles)"""
        # Add 1-2 stress sources that will propagate pathology
        num_stress_sources = max(1, self.rows // 3)
        
        for _ in range(num_stress_sources):
            # Place stress sources in vulnerable regions
            stress_r = np.random.choice([1, 2])  # Layer 2/3 or Layer 5 (vulnerable)
            stress_c = np.random.randint(0, self.cols)
            
            self.stress_sources.append((stress_r, stress_c))
            
            # Apply pathological stress to neuron and neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = stress_r + dr, stress_c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        neuron = self.neurons[nr][nc]
                        stress_intensity = 1.0 if (dr == 0 and dc == 0) else 0.5
                        neuron.apply_pathological_stress(stress_intensity)

    def _get_neuron_at_grid_pos(self, r: int, c: int):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.neurons[r][c]
        return None
    
    def _create_connection(self, pre_neuron, post_neuron, pre_r, pre_c, post_r, post_c):
        """Create synaptic connection with biological constraints"""
        conn_id = f"{pre_neuron.name}_to_{post_neuron.name}"

        if conn_id in self._created_unidirectional_connections:
            return False
        
        # Biological constraint: connection probability based on distance and layer
        connection_prob = self._calculate_connection_probability(pre_r, pre_c, post_r, post_c)
        if np.random.random() > connection_prob:
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
            
            # Initial weight based on pre-synaptic neuron health
            initial_weight = self.initial_syn_weight * (1.0 - pre_neuron.vulnerability * 0.3)
            
            netcon_to_synapse = h.NetCon(pre_neuron.soma_segment.hoc._ref_v, 
                                       point_process_mech.hoc, 
                                       sec=pre_neuron.soma.hoc)
            netcon_to_synapse.weight[0] = initial_weight
            netcon_to_synapse.delay = 1.0 + pre_neuron.vulnerability  # Slower in unhealthy neurons
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

    def _calculate_connection_probability(self, pre_r, pre_c, post_r, post_c):
        """Calculate biologically realistic connection probability"""
        distance = np.sqrt((pre_r - post_r)**2 + (pre_c - post_c)**2)
        
        # Distance-dependent probability (closer = more likely)
        distance_prob = np.exp(-distance / 2.0)
        
        # Layer-specific connectivity rules
        pre_layer = self._get_cortical_layer(pre_r, pre_c)
        post_layer = self._get_cortical_layer(post_r, post_c)
        
        layer_prob = 1.0
        if pre_layer == "Layer2/3" and post_layer == "Layer5":
            layer_prob = 0.8  # Strong cortico-cortical connections
        elif pre_layer == "Layer4":
            layer_prob = 0.9  # Input layer connects widely
        elif distance > 2:
            layer_prob = 0.3  # Long-range connections less likely
            
        return distance_prob * layer_prob

    def _build_grid_connections(self):
        """Build biologically realistic connectivity"""
        if self.num_neurons < 2:
            print("Need at least 2 neurons")
            return

        connection_count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                pre_neuron = self._get_neuron_at_grid_pos(r, c)
                
                # Connect to nearby neurons (biological constraint)
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

    def biological_synaptic_plasticity(self):
        """Biologically accurate synaptic plasticity and pruning"""
        pruned_connections = []
        
        for i, conn_info in enumerate(self.connections):
            if conn_info['is_pruned']:
                continue
                
            pre_neuron = conn_info['pre_neuron']
            post_neuron = conn_info['post_neuron']
            netcon_to_synapse = conn_info['netcon_to_synapse']

            # Get biochemical states
            pre_growth = pre_neuron.ode_mech.growth_strength
            pre_apop = pre_neuron.ode_mech.apop_strength
            post_growth = post_neuron.ode_mech.growth_strength
            post_apop = post_neuron.ode_mech.apop_strength
            
            # Calculate net states
            pre_state = pre_growth - pre_apop
            post_state = post_growth - post_apop
            
            current_weight = netcon_to_synapse.weight[0]
            
            # Biological plasticity rules
            if pre_state > 0.3 and post_state > 0.3:
                # Both neurons healthy and active - strengthen (LTP-like)
                weight_change = self.learning_rate * 2.0 * min(pre_state, post_state)
            elif pre_state < -0.2 or post_state < -0.2:
                # One or both neurons stressed - weaken (LTD-like)
                weight_change = -self.learning_rate * 3.0 * max(abs(pre_state), abs(post_state))
            elif abs(pre_state) < 0.1 and abs(post_state) < 0.1:
                # Both neurons silent - activity-dependent pruning
                weight_change = -self.learning_rate * 0.5
            else:
                # Maintenance
                weight_change = 0
            
            # Apply homeostatic scaling
            if current_weight > 0.8:
                weight_change *= 0.5  # Reduce growth for strong synapses
            elif current_weight < 0.2:
                weight_change *= 1.5  # Boost weak synapses or prune them
            
            new_weight = current_weight + weight_change
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            
            # Synaptic pruning
            if new_weight < self.prune_threshold:
                conn_info['is_pruned'] = True
                netcon_to_synapse.weight[0] = 0.0
                pruned_connections.append(i)
                print(f"Pruned: {pre_neuron.name} -> {post_neuron.name}")
            else:
                netcon_to_synapse.weight[0] = new_weight

        return len(pruned_connections)

    def propagate_pathological_stress(self, current_time):
        """Simulate spreading of pathological stress through the network"""
        if current_time < 100:  # Start pathology after 100ms
            return
            
        # Pathology spreads through connections
        for conn_info in self.connections:
            if conn_info['is_pruned']:
                continue
                
            pre_neuron = conn_info['pre_neuron']
            post_neuron = conn_info['post_neuron']
            
            # If pre-neuron is highly stressed, spread to post-neuron
            if pre_neuron.pathological_stress > 0.5:
                stress_transfer = 0.001 * pre_neuron.pathological_stress * conn_info['netcon_to_synapse'].weight[0]
                post_neuron.accumulate_stress(stress_transfer)

    def get_network_health_metrics(self):
        """Calculate network-wide health metrics"""
        total_neurons = self.rows * self.cols
        active_connections = sum(1 for conn in self.connections if not conn['is_pruned'])
        
        avg_growth = np.mean([n.ode_mech.growth_strength for row in self.neurons for n in row])
        avg_apop = np.mean([n.ode_mech.apop_strength for row in self.neurons for n in row])
        avg_stress = np.mean([n.pathological_stress for row in self.neurons for n in row])
        
        return {
            'active_connections': active_connections,
            'connection_density': active_connections / (total_neurons * (total_neurons - 1)),
            'avg_growth_signal': avg_growth,
            'avg_apoptosis_signal': avg_apop,
            'avg_pathological_stress': avg_stress,
            'network_health': avg_growth - avg_apop - avg_stress
        }

    def visualize_biological_network(self, time_step, current_time, save_fig=False):
        """Visualize network with biological states"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        # Calculate neuron positions
        neuron_positions = {}
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * 3.0
                y = (self.rows - r - 1) * 3.0
                neuron_positions[(r, c)] = (x, y)
        
        # Draw connections with pruning visualization
        for conn in self.connections:
            if conn['is_pruned']:
                continue  # Don't draw pruned connections
                
            if all(k in conn for k in ['pre_r', 'pre_c', 'post_r', 'post_c']):
                pre_pos = neuron_positions[(conn['pre_r'], conn['pre_c'])]
                post_pos = neuron_positions[(conn['post_r'], conn['post_c'])]
                
                weight = conn['netcon_to_synapse'].weight[0]
                thickness = max(0.3, min(6.0, weight * 8))
                
                # Color based on health
                if weight > 0.7:
                    color = 'darkgreen'
                elif weight > 0.3:
                    color = 'orange'
                else:
                    color = 'red'
                
                ax.plot([pre_pos[0], post_pos[0]], [pre_pos[1], post_pos[1]], 
                       color=color, linewidth=thickness, alpha=0.6)
        
        # Draw neurons with biological states
        for r in range(self.rows):
            for c in range(self.cols):
                neuron = self.neurons[r][c]
                pos = neuron_positions[(r, c)]
                
                # Neuron color based on health state
                growth = neuron.ode_mech.growth_strength
                apop = neuron.ode_mech.apop_strength
                stress = neuron.pathological_stress
                
                net_health = growth - apop - stress
                
                if net_health > 0.3:
                    neuron_color = 'lightgreen'    # Healthy
                elif net_health > -0.2:
                    neuron_color = 'yellow'        # Stressed
                elif net_health > -0.5:
                    neuron_color = 'orange'        # Declining
                else:
                    neuron_color = 'red'           # Dying/dead
                
                # Size based on activity and health
                activity = neuron.ode_mech.activity_level
                size = max(200, min(1500, 500 + activity * 2 - stress * 300))
                
                # Special marking for stress sources
                if (r, c) in self.stress_sources:
                    edge_color = 'purple'
                    edge_width = 6
                else:
                    edge_color = 'black'
                    edge_width = 2
                
                ax.scatter(pos[0], pos[1], s=size, c=neuron_color, 
                          edgecolors=edge_color, linewidth=edge_width, alpha=0.8)
                
                # Label with biological info
                ax.text(pos[0], pos[1], f'({r},{c})\nH:{net_health:.1f}\nS:{stress:.1f}',
                       ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Mark cortical layers
        for r in range(self.rows):
            layer = self._get_cortical_layer(r, 0)
            ax.text(-2, (self.rows - r - 1) * 3, layer, rotation=90, 
                   ha='center', va='center', fontsize=10, alpha=0.7)
        
        ax.set_xlim(-3, self.cols * 3)
        ax.set_ylim(-1, self.rows * 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Biological Neural Network - t={current_time:.1f}ms\nNeurodegeneration & Synaptic Pruning', 
                    fontsize=14)
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        
        # Legend
        legend_elements = [
            plt.scatter([], [], s=300, c='lightgreen', edgecolors='black', label='Healthy'),
            plt.scatter([], [], s=300, c='yellow', edgecolors='black', label='Stressed'),
            plt.scatter([], [], s=300, c='orange', edgecolors='black', label='Declining'),
            plt.scatter([], [], s=300, c='red', edgecolors='black', label='Dying'),
            plt.scatter([], [], s=300, c='gray', edgecolors='purple', linewidth=4, label='Pathology Source'),
            plt.Line2D([0], [0], color='darkgreen', lw=4, label='Strong Synapse'),
            plt.Line2D([0], [0], color='orange', lw=4, label='Weak Synapse'),
            plt.Line2D([0], [0], color='red', lw=4, label='Failing Synapse')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'biological_network_t{time_step:04d}.png', dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.1)

class BiologicalODENeuron(ODENeuron):
    """Extended ODENeuron with biological stress and pathology"""
    
    def __init__(self, name, initial_concentrations, params, grid_pos=(0,0), vulnerability=0.5, cortical_layer="Layer4"):
        super().__init__(name, initial_concentrations, params)
        
        self.grid_pos = grid_pos
        self.vulnerability = vulnerability
        self.cortical_layer = cortical_layer
        self.pathological_stress = 0.0
        self.cumulative_damage = 0.0
        
        print(f"BiologicalODENeuron '{self.name}' initialized - Layer: {cortical_layer}, Vulnerability: {vulnerability:.2f}")
    
    def apply_pathological_stress(self, stress_intensity):
        """Apply initial pathological stress (e.g., from amyloid plaques)"""
        self.pathological_stress += stress_intensity * self.vulnerability
        
        # Immediately affect biochemistry
        if self.pathological_stress > 0.3:
            # Increase proBDNF production under stress
            self.ode_mech.ksP *= (1.0 + self.pathological_stress)
            # Decrease TrkB receptors
            self.ode_mech.ks_TrkB *= (1.0 - self.pathological_stress * 0.5)
    
    def accumulate_stress(self, stress_amount):
        """Accumulate stress over time from network pathology"""
        self.pathological_stress += stress_amount * self.vulnerability
        self.cumulative_damage += stress_amount
        
        # Progressive damage affects synthesis rates
        if self.cumulative_damage > 0.1:
            damage_factor = min(0.8, self.cumulative_damage)
            self.ode_mech.ks_TrkB *= (1.0 - damage_factor)
            self.ode_mech.ks_p75 *= (1.0 + damage_factor)
    
    def calculate_and_get_neuron_state(self) -> float:
        """Enhanced state calculation including pathological stress"""
        growth_strength = self.ode_mech.growth_strength
        apop_strength = self.ode_mech.apop_strength
        
        # Include pathological stress in state calculation
        signal = growth_strength - apop_strength - self.pathological_stress
        self.neuron_state = signal
        return self.neuron_state


# Example usage
# Complete the main function for comprehensive testing and analysis

if __name__ == "__main__":
    print("=== Biologically Accurate Neural Network Simulation ===")
    
    # Load mechanisms (same as before)
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
    
    # Parameters for biological realism
    initial_neuron_concentrations = [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    base_neuron_parameters = [
        5.0e-3, 0.01, 1.0, 0.9, 5.0e-4, 0.2, 0.1, 1.0, 0.9, 0.005, 0.3, 0.1, 
        0.0001, 0.00001, 0.0005, 0.0005, 0.0005, 0.0005, 0.9, 0.1, 0.1, 0.9, 
        0.0011, 0.001, 0.002, 0.003, 0.004, 1.0
    ]
    
    # Create biologically accurate network
    print("\n🧬 Creating Biological Neural Network...")
    network = BiologicalNeuralNetwork(
        rows=5, 
        cols=6, 
        initial_neuron_concentrations=initial_neuron_concentrations,
        base_neuron_params=base_neuron_parameters,
        synapse_type="ProbabilisticSyn",
        initial_syn_weight=0.6,
        learning_rate=0.002,
        min_weight=0.01,
        max_weight=1.0,
        prune_threshold=0.08
    )
    
    # Display initial network state
    print(f"\n📊 Initial Network Statistics:")
    initial_metrics = network.get_network_health_metrics()
    for key, value in initial_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Analysis storage
    time_points = []
    health_metrics = []
    neuron_states = []
    connection_counts = []
    
    # Simulation parameters
    h.dt = 0.1
    simulation_time = 500.0  # 500ms
    plasticity_interval = 10.0  # Update plasticity every 10ms
    analysis_interval = 25.0   # Save analysis every 25ms
    visualization_interval = 100.0  # Visualize every 100ms
    
    print(f"\n🔬 Starting Simulation (t=0 to {simulation_time}ms)")
    print("=" * 60)
    
    # Initialize NEURON
    h.finitialize(-65)
    
    current_time = 0.0
    time_step = 0
    last_plasticity_update = 0.0
    last_analysis_save = 0.0
    last_visualization = 0.0
    
    # Create output directory for visualizations
    os.makedirs("biological_network_analysis", exist_ok=True)
    
    while current_time <= simulation_time:
        # Run NEURON simulation step
        h.fadvance()
        current_time = h.t
        time_step += 1
        
        # Apply biological processes
        if current_time - last_plasticity_update >= plasticity_interval:
            # Synaptic plasticity and pruning
            pruned_count = network.biological_synaptic_plasticity()
            
            # Pathological stress propagation
            network.propagate_pathological_stress(current_time)
            
            last_plasticity_update = current_time
            
            if pruned_count > 0:
                print(f"t={current_time:.1f}ms: Pruned {pruned_count} synapses")
        
        # Collect analysis data
        if current_time - last_analysis_save >= analysis_interval:
            metrics = network.get_network_health_metrics()
            time_points.append(current_time)
            health_metrics.append(metrics)
            
            # Collect individual neuron states
            states = []
            for r in range(network.rows):
                row_states = []
                for c in range(network.cols):
                    neuron = network.neurons[r][c]
                    neuron_info = {
                        'position': (r, c),
                        'growth_strength': neuron.ode_mech.growth_strength,
                        'apop_strength': neuron.ode_mech.apop_strength,
                        'pathological_stress': neuron.pathological_stress,
                        'vulnerability': neuron.vulnerability,
                        'layer': neuron.cortical_layer,
                        'activity': neuron.ode_mech.activity_level,
                        'net_health': neuron.ode_mech.growth_strength - neuron.ode_mech.apop_strength - neuron.pathological_stress
                    }
                    row_states.append(neuron_info)
                states.append(row_states)
            neuron_states.append(states)
            
            connection_counts.append(metrics['active_connections'])
            last_analysis_save = current_time
            
            # Progress report
            if int(current_time) % 50 == 0:
                print(f"t={current_time:.1f}ms - Health: {metrics['network_health']:.3f}, "
                      f"Connections: {metrics['active_connections']}, "
                      f"Avg Stress: {metrics['avg_pathological_stress']:.3f}")
        
        # Visualization
        if current_time - last_visualization >= visualization_interval:
            print(f"\n📸 Generating visualization at t={current_time:.1f}ms")
            network.visualize_biological_network(
                time_step, current_time, 
                save_fig=True
            )
            last_visualization = current_time
    
    print(f"\n✅ Simulation Complete!")
    print("=" * 60)
    
    # Comprehensive Analysis
    print("\n📈 COMPREHENSIVE ANALYSIS")
    print("=" * 40)
    
    # 1. Network Health Evolution
    final_metrics = health_metrics[-1]
    initial_metrics = health_metrics[0]
    
    print(f"\n🔍 Network Health Evolution:")
    print(f"  Initial Health: {initial_metrics['network_health']:.4f}")
    print(f"  Final Health:   {final_metrics['network_health']:.4f}")
    print(f"  Health Change:  {final_metrics['network_health'] - initial_metrics['network_health']:.4f}")
    
    print(f"\n🔗 Connectivity Changes:")
    print(f"  Initial Connections: {initial_metrics['active_connections']}")
    print(f"  Final Connections:   {final_metrics['active_connections']}")
    print(f"  Pruned Connections:  {initial_metrics['active_connections'] - final_metrics['active_connections']}")
    print(f"  Pruning Rate:        {((initial_metrics['active_connections'] - final_metrics['active_connections']) / initial_metrics['active_connections'] * 100):.1f}%")
    
    # 2. Layer-wise Analysis
    print(f"\n🧠 Layer-wise Vulnerability Analysis:")
    final_states = neuron_states[-1]
    layer_stats = {}
    
    for r in range(network.rows):
        for c in range(network.cols):
            neuron_info = final_states[r][c]
            layer = neuron_info['layer']
            
            if layer not in layer_stats:
                layer_stats[layer] = {
                    'count': 0,
                    'avg_health': 0,
                    'avg_stress': 0,
                    'avg_vulnerability': 0,
                    'surviving_neurons': 0
                }
            
            layer_stats[layer]['count'] += 1
            layer_stats[layer]['avg_health'] += neuron_info['net_health']
            layer_stats[layer]['avg_stress'] += neuron_info['pathological_stress']
            layer_stats[layer]['avg_vulnerability'] += neuron_info['vulnerability']
            
            if neuron_info['net_health'] > -0.3:  # Survival threshold
                layer_stats[layer]['surviving_neurons'] += 1
    
    for layer, stats in layer_stats.items():
        count = stats['count']
        survival_rate = (stats['surviving_neurons'] / count) * 100
        print(f"  {layer}:")
        print(f"    Health: {stats['avg_health']/count:.3f}")
        print(f"    Stress: {stats['avg_stress']/count:.3f}")
        print(f"    Vulnerability: {stats['avg_vulnerability']/count:.3f}")
        print(f"    Survival Rate: {survival_rate:.1f}%")
    
    # 3. Pathology Spread Analysis
    print(f"\n🦠 Pathological Stress Propagation:")
    stress_sources = network.stress_sources
    print(f"  Stress Sources: {stress_sources}")
    
    for source_r, source_c in stress_sources:
        print(f"\n  Source at ({source_r}, {source_c}):")
        # Analyze spread from this source
        for radius in [1, 2, 3]:
            neurons_in_radius = []
            for r in range(max(0, source_r-radius), min(network.rows, source_r+radius+1)):
                for c in range(max(0, source_c-radius), min(network.cols, source_c+radius+1)):
                    if abs(r-source_r) <= radius and abs(c-source_c) <= radius:
                        neuron_info = final_states[r][c]
                        neurons_in_radius.append(neuron_info['pathological_stress'])
            
            if neurons_in_radius:
                avg_stress = np.mean(neurons_in_radius)
                print(f"    Radius {radius}: Avg stress = {avg_stress:.3f}")
    
    # 4. Generate Analysis Plots
    print(f"\n📊 Generating Analysis Plots...")
    
    # Plot 1: Network Health Over Time
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 3, 1)
    times = np.array(time_points)
    health_values = [m['network_health'] for m in health_metrics]
    plt.plot(times, health_values, 'b-', linewidth=2, label='Network Health')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time (ms)')
    plt.ylabel('Network Health')
    plt.title('Network Health Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Connection Pruning
    plt.subplot(2, 3, 2)
    plt.plot(times, connection_counts, 'g-', linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Active Connections')
    plt.title('Synaptic Pruning Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Growth vs Apoptosis Signals
    plt.subplot(2, 3, 3)
    growth_values = [m['avg_growth_signal'] for m in health_metrics]
    apop_values = [m['avg_apoptosis_signal'] for m in health_metrics]
    plt.plot(times, growth_values, 'g-', linewidth=2, label='Growth Signal')
    plt.plot(times, apop_values, 'r-', linewidth=2, label='Apoptosis Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal Strength')
    plt.title('Growth vs Apoptosis Signals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Pathological Stress Evolution
    plt.subplot(2, 3, 4)
    stress_values = [m['avg_pathological_stress'] for m in health_metrics]
    plt.plot(times, stress_values, 'purple', linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Pathological Stress')
    plt.title('Pathological Stress Propagation')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Layer Survival Rates
    plt.subplot(2, 3, 5)
    layers = list(layer_stats.keys())
    survival_rates = [(layer_stats[layer]['surviving_neurons'] / layer_stats[layer]['count']) * 100 
                     for layer in layers]
    colors = ['lightblue', 'orange', 'lightgreen', 'salmon', 'lightcoral']
    plt.bar(layers, survival_rates, color=colors[:len(layers)])
    plt.xlabel('Cortical Layer')
    plt.ylabel('Survival Rate (%)')
    plt.title('Layer-wise Neuron Survival')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Network Connectivity Density
    plt.subplot(2, 3, 6)
    density_values = [m['connection_density'] for m in health_metrics]
    plt.plot(times, density_values, 'orange', linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Connection Density')
    plt.title('Network Connectivity Density')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('biological_network_analysis/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Save Analysis Data
    print(f"\n💾 Saving Analysis Data...")
    
    # Save time series data
    analysis_data = {
        'time_points': time_points,
        'health_metrics': health_metrics,
        'neuron_states': neuron_states,
        'layer_stats': layer_stats,
        'stress_sources': stress_sources,
        'simulation_params': {
            'rows': network.rows,
            'cols': network.cols,
            'simulation_time': simulation_time,
            'initial_connections': initial_metrics['active_connections'],
            'final_connections': final_metrics['active_connections']
        }
    }
    
    import pickle
    with open('biological_network_analysis/analysis_data.pkl', 'wb') as f:
        pickle.dump(analysis_data, f)
    
    print(f"✅ Analysis data saved to 'biological_network_analysis/'")
    
    # 6. Summary Report
    print(f"\n📋 SIMULATION SUMMARY REPORT")
    print("=" * 50)
    print(f"🕒 Simulation Duration: {simulation_time}ms")
    print(f"🧠 Network Size: {network.rows}×{network.cols} = {network.num_neurons} neurons")
    print(f"🔗 Initial Synapses: {initial_metrics['active_connections']}")
    print(f"✂️  Synapses Pruned: {initial_metrics['active_connections'] - final_metrics['active_connections']}")
    print(f"📉 Network Health Decline: {(final_metrics['network_health'] - initial_metrics['network_health']):.3f}")
    print(f"🦠 Final Pathological Stress: {final_metrics['avg_pathological_stress']:.3f}")
    
    most_vulnerable_layer = min(layer_stats.keys(), 
                               key=lambda l: layer_stats[l]['surviving_neurons']/layer_stats[l]['count'])
    most_resilient_layer = max(layer_stats.keys(), 
                              key=lambda l: layer_stats[l]['surviving_neurons']/layer_stats[l]['count'])
    
    print(f"🚨 Most Vulnerable Layer: {most_vulnerable_layer}")
    print(f"🛡️  Most Resilient Layer: {most_resilient_layer}")
    
    print(f"\n🔬 Key Biological Features Demonstrated:")
    print(f"  ✓ Localized biochemical differentiation")
    print(f"  ✓ Layer-specific vulnerability patterns")
    print(f"  ✓ Activity-dependent synaptic pruning")
    print(f"  ✓ Pathological stress propagation")
    print(f"  ✓ Neurodegeneration cascades")
    print(f"  ✓ Homeostatic plasticity mechanisms")
    
    print(f"\n🎯 Files Generated:")
    print(f"  • Network visualizations: biological_network_t*.png")
    print(f"  • Comprehensive analysis: comprehensive_analysis.png")
    print(f"  • Analysis data: analysis_data.pkl")
    
    print(f"\n✨ Biological Neural Network Simulation Complete! ✨")