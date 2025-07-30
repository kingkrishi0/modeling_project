from neuron import h, nrn
from neuronpp.cells.cell import Cell
from neuronpp.core.hocwrappers.synapses.single_synapse import SingleSynapse
import numpy as np
from neuronpp.core.hocwrappers.point_process import PointProcess
import os
from ode_neuron_class import ODENeuron

class NeuronalNetwork:
    def __init__(self, rows: int, cols: int, initial_neuron_concentrations: list, neuron_params: list,
                 synapse_type: str = "ExpSyn",
                 initial_syn_weight: float = 0.01,
                 learning_rate: float = 0.0001,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 threshold_growth: float = 0.2,
                 threshold_apoptosis: float = -0.2):
        
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

        print(f"Building neuronal network")

        for r in range(self.rows):
            newrow = []
            for c in range(self.cols):
                neuron = ODENeuron(name=f'neuron{r}{c}', initial_concentrations=initial_neuron_concentrations, 
                                params=neuron_params)
                newrow.append(neuron)
            self.neurons.append(newrow)


        self._build_grid_connections()
        print(f"Network made awesome!")

    def _get_neuron_at_grid_pos(self, r: int, c: int):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.neurons[r][c]
        return None
    
    def _create_connection(self, pre_neuron, post_neuron):
        
        conn_id = f"{pre_neuron.name}_to_{post_neuron.name}"

        if conn_id in self._created_unidirectional_connections:
            return False
        
        pre_segment_loc = pre_neuron.soma(0.5) # neuronpp.core.hocwrappers.sec.Seg object
        post_segment_loc = post_neuron.soma(0.5) # The specific postsynaptic Segment for insertion

        # 1. Get the raw h.Section object from the neuronpp.Seg wrapper's parent (which is a Sec)
        #    post_segment_loc.parent is the neuronpp.Sec (self.soma for post_neuron).
        #    post_segment_loc.parent.hoc is the raw h.Section.
        #    (post_segment_loc.parent.hoc)(post_segment_loc.x) gets the h.Section at the segment's x.
        h_section_at_segment = (post_segment_loc.parent.hoc)(post_segment_loc.x)

        # 2. Directly create the NEURON PointProcess (e.g., h.ExpSyn) on the h.Section.
        #    This is the lowest-level, most direct way to instantiate a Point Process.
        #    We need to map the synapse_type string to the actual h.mechanism_name constructor.
        if hasattr(h, self.synapse_type): # Check if h.ExpSyn, h.Synapses, etc. exists
            raw_h_point_process = getattr(h, self.synapse_type)(h_section_at_segment)
        else:
            raise ValueError(f"Synapse type '{self.synapse_type}' is not a recognized NEURON PointProcess. "
                             f"Ensure it's compiled and correct. (e.g., ExpSyn)")

        # 3. Wrap this raw h.PointProcess object with neuronpp's PointProcess wrapper.
        #    This is needed because SingleSynapse expects a neuronpp.PointProcess object.
        point_process_mech = PointProcess(
            hoc_obj=raw_h_point_process,
            parent=post_segment_loc.parent, # The neuronpp.Sec object
            name=f"{self.synapse_type}_{conn_id}",
            cell = post_neuron, # Give it a unique name
            mod_name = self.synapse_type # This is the mechanism name, e.g., ExpSyn
        )
        
        syn_obj = SingleSynapse(
            source=pre_segment_loc, 
            point_process=point_process_mech, 
            name=conn_id 
           
        )

        syn_obj.add_netcon(source=pre_segment_loc, weight=self.initial_syn_weight, delay=1.0) # Or self.initial_syn_delay if you add it to init


        self.connections.append({
            'pre_neuron': pre_neuron,
            'post_neuron': post_neuron,
            'syn': syn_obj # This is the SingleSynapse object
        })

        self._created_unidirectional_connections.add(conn_id)
        print(f"Connection established: {pre_neuron.name} -> {post_neuron.name}")
        
        return True

    def _build_grid_connections(self):
        if self.num_neurons < 2:
            print("Need at least 2 neurons")
            return

        for r in range(self.rows):
            for c in range(self.cols):
                pre_neuron = self._get_neuron_at_grid_pos(r, c)
                offsets = [(0,1), (1, 0)]

                for dr, dc in offsets:
                    post_r, post_c = r+dr, c+dc

                    post_neuron = self._get_neuron_at_grid_pos(post_r, post_c)

                    if post_neuron:
                        self._create_connection(pre_neuron, post_neuron)
                        self._create_connection(post_neuron, pre_neuron)

    def modulate_synaptic_weights(self):
        for conn_info in self.connections:
            pre_neuron = conn_info['pre_neuron']
            post_neuron = conn_info['post_neuron']
            syn_obj = conn_info['syn']

            pre_n_state = pre_neuron.calculate_and_get_neuron_state()
            post_n_state = post_neuron.calculate_and_get_neuron_state()

            current_weight = syn_obj.netcons[0].hoc.weight[0]
            new_weight = current_weight

            if pre_n_state > self.threshold_growth and post_n_state > self.threshold_growth:
                new_weight += self.learning_rate


            elif pre_n_state < self.threshold_apoptosis or post_n_state < self.threshold_apoptosis:
                new_weight -= self.learning_rate

            new_weight = max(self.min_weight, new_weight)
            new_weight = min(self.max_weight, new_weight)

            syn_obj.netcons[0].hoc.weight[0] = new_weight
# Assuming ODENeuron and NeuronalNetwork classes are defined above this point
# and neuronpp imports are correct.

if __name__ == "__main__":
    print("--- Starting Simulation Script ---")

    # --- 1. Compile NMODL files ---
    mod_dir = os.path.join(os.path.dirname(__file__), 'mods')
    if os.path.exists(mod_dir):
        print(f"Compiling NMODL files in {mod_dir}...")
        os.system(f"nrnivmodl {mod_dir}")
    else:
        print("Warning: 'mods' directory not found. Assuming ode_neuron.mod is in the current directory.")
        print("Attempting to compile NMODL files in current directory...")
        os.system("nrnivmodl") # Compile mods in current directory

    # Import the compiled mechanism after compilation
    h.nrn_load_dll(os.path.join("x86_64", ".libs", "libnrnmech.so"))# Exit if mechanisms can't be loaded

    # --- 2. Define Initial Conditions and Parameters ---
    # These must match the expected input for your ODENeuron.__init__
    initial_neuron_concentrations = [
        0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1
    ]
    neuron_parameters = [
        5.0e-3, 0.01, 1.0, 0.9, 5.0e-4, 0.2, 0.1, 1.0, 0.9, 0.005, 0.3, 0.2, 0.0001, 0.00001,
        0.0005, 0.0005, 0.0005, 0.0005, 0.9, 0.1, 0.1, 0.9, 0.0011, 0.0001, 0.0001, 0.00001
    ]

    # --- 3. Create the Neuronal Network Instance ---
    grid_rows = 2 # Example: 2 rows
    grid_cols = 2 # Example: 2 columns (total 4 neurons)
    network = NeuronalNetwork(rows=grid_rows, cols=grid_cols,
                              initial_neuron_concentrations=initial_neuron_concentrations,
                              neuron_params=neuron_parameters,
                              synapse_type="ExpSyn", # Ensure ExpSyn.mod is compiled
                              initial_syn_weight=0.5,
                              learning_rate=0.0005, # Adjusted for potentially smoother changes
                              min_weight=0.0,
                              max_weight=1.0)

    print(f"\n--- Network Initialization Summary ---")
    print(f"Network Dimensions: {network.rows}x{network.cols}")
    print(f"Total Neurons Created: {len(network.neurons)}")
    print(f"Total Bidirectional Connections: {len(network.connections)}")

    # --- 4. Verify Connections and Initial States (Testing Statements) ---
    if len(network.neurons) > 0:
        first_neuron = network.neurons[0][0]
        print(f"\nInitial State of {first_neuron.name}: {first_neuron.calculate_and_get_neuron_state():.4f}")
        print(f"Initial ProBDNF (P) of {first_neuron.name}: {first_neuron.P[0]:.4e} M")

    if len(network.connections) > 0:
        first_connection_info = network.connections[0]
        pre_n = first_connection_info['pre_neuron']
        post_n = first_connection_info['post_neuron']
        syn = first_connection_info['syn'] # 'syn' is the SingleSynapse object

        print(f"\nExample Connection: {pre_n.name} -> {post_n.name}")

        # --- FIX: Access all NetCon attributes directly via the .hoc attribute ---
        # This is the most reliable way if get_methods are absent.
        netcon_hoc = syn.netcons[0].hoc # Get the raw h.NetCon object once for this synapse

        print(f"  Initial Weight: {netcon_hoc.weight[0]:.4f}") # h.NetCon.weight is a 1-element vector
        print(f"  Synapse Type: {syn.point_process_name}")
        print(f"  Synapse Delay: {netcon_hoc.delay:.4f}") # h.NetCon.delay is a direct float

        try:
            # Source and target on h.NetCon (netcon_hoc) are typically h.Segment objects
            print(f"  Pre-synaptic Segment: {netcon_hoc.source.sec.name()}({netcon_hoc.source.x:.2f})")
            print(f"  Post-synaptic Segment: {netcon_hoc.target.sec.name()}({netcon_hoc.target.x:.2f})")
        except AttributeError:
            print("  Could not retrieve segment names via .hoc.source/target (unexpected structure).")
    else:
        print("\nNo connections created in the network.")

    # --- 5. Setup NEURON simulation parameters ---
    h.finitialize(-65) # Initialize membrane potential
    h.t = 0
    h.dt = 0.01 # seconds (10 ms)

    # --- 6. Recording Variables (for Plotting) ---
    # Store data in standard Python lists for easy conversion to NumPy arrays
    t_list = []


    # These lists will store [row][col] -> list_of_values
    all_P_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_B_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_growth_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_apop_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_state_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_activity_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_p75_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_TrkB_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_p75_pro_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_p75_B_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)] # Corrected variable name
    all_TrkB_B_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)] # Corrected variable name
    all_TrkB_pro_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_tPA_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]


    # Recording for a specific synapse's weight (e.g., neuron_r0c0 -> neuron_r0c1)
    # Find the target synapse once before the loop to avoid re-searching
    target_synapse_for_plotting = None
    if grid_cols > 1: # Ensure there's a neighbor to connect to
        # Assuming neuron at (0,0) connects to neuron at (0,1)
        # This will be network.neurons[0] to network.neurons[1]
        for conn in network.connections:
            if conn['pre_neuron'].name == "neuron_r0c0" and conn['post_neuron'].name == "neuron_r0c1":
                target_synapse_for_plotting = conn['syn']
                break
    if not target_synapse_for_plotting and len(network.connections) > 0:
        target_synapse_for_plotting = network.connections[0]['syn'] # Fallback to first available
        print("Note: Plotting the weight of the first available connection.")
    elif not target_synapse_for_plotting:
        print("Note: No synapses found to plot weights.")

    syn_weights_list = []


    # --- Simulation loop ---
    runtime = 2000 # seconds
    num_steps = int(runtime / h.dt)

    burst_duration_sec = 500  # seconds
    inter_burst_interval_sec = 1000 # seconds
    high_activity_value = 2.0
    low_activity_value = 0.01

    print("\n--- Starting Simulation Loop ---")
    for i in range(num_steps):
        current_time = h.t

        # --- FIX: Determine activity for ALL neurons based on burst pattern ---
        current_activity_for_all = low_activity_value 
        time_within_cycle = current_time % inter_burst_interval_sec
        if time_within_cycle < burst_duration_sec:
            current_activity_for_all = high_activity_value
        
        # Iterate through ALL neurons to update their activity and append their data
        for r_idx in range(network.rows):
            for c_idx in range(network.cols):
                neuron_obj = network.neurons[r_idx][c_idx]

                # Update activity for the current neuron in the loop
                neuron_obj.update_activity_level(current_activity_for_all) 

                # --- Data Appending for EACH neuron (already correctly using r_idx, c_idx) ---
                all_P_data[r_idx][c_idx].append(neuron_obj.P[0])
                all_B_data[r_idx][c_idx].append(neuron_obj.B[0])
                all_growth_data[r_idx][c_idx].append(neuron_obj.growth_strength_ref[0])
                all_apop_data[r_idx][c_idx].append(neuron_obj.apop_strength_ref[0])
                all_state_data[r_idx][c_idx].append(neuron_obj.calculate_and_get_neuron_state())
                all_activity_data[r_idx][c_idx].append(neuron_obj.activity_level_ref[0])

                all_p75_data[r_idx][c_idx].append(neuron_obj.p75[0])
                all_TrkB_data[r_idx][c_idx].append(neuron_obj.TrkB[0])
                all_p75_pro_data[r_idx][c_idx].append(neuron_obj.p75_pro[0])
                all_p75_B_data[r_idx][c_idx].append(neuron_obj.p75_B[0])
                all_TrkB_B_data[r_idx][c_idx].append(neuron_obj.TrkB_B[0])
                all_TrkB_pro_data[r_idx][c_idx].append(neuron_obj.TrkB_pro[0])
                all_tPA_data[r_idx][c_idx].append(neuron_obj.tPA[0])

        h.fadvance() # Advance simulation once per time step, *after* all neurons' inputs are set

        t_list.append(h.t) # Time list is still 1D

        if target_synapse_for_plotting:
            syn_weights_list.append(target_synapse_for_plotting.netcons[0].hoc.weight[0])


        # Apply Synaptic Weight Modulation (after all neurons have potentially updated their states)
        network.modulate_synaptic_weights()

    print("--- Simulation Finished ---")

    # --- 9. Convert lists to numpy arrays for plotting ---
    times = np.array(t_list)
    all_P_data_np = [[np.array(all_P_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_B_data_np = [[np.array(all_B_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_growth_data_np = [[np.array(all_growth_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_apop_data_np = [[np.array(all_apop_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_state_data_np = [[np.array(all_state_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_activity_data_np = [[np.array(all_activity_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_p75_data_np = [[np.array(all_p75_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_TrkB_data_np = [[np.array(all_TrkB_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_p75_pro_data_np = [[np.array(all_p75_pro_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_p75_B_data_np = [[np.array(all_p75_B_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_TrkB_B_data_np = [[np.array(all_TrkB_B_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_TrkB_pro_data_np = [[np.array(all_TrkB_pro_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_tPA_data_np = [[np.array(all_tPA_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]

    syn_weights_data = np.array(syn_weights_list) # This remains for the single synapse plot



    # --- 10. Plotting Results ---
    import matplotlib.pyplot as plt

    for r_plot in range(network.rows):
        for c_plot in range(network.cols):
            neuron_name = network.neurons[r_plot][c_plot].name
            
            # Create a NEW FIGURE for each neuron
            fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True) # 5 subplots per neuron, share time axis
            fig.suptitle(f'Dynamics for Neuron: {neuron_name}', fontsize=16)

            # Access data for the current neuron (r_plot, c_plot)
            P_data_curr = all_P_data_np[r_plot][c_plot]
            B_data_curr = all_B_data_np[r_plot][c_plot]
            activity_data_curr = all_activity_data_np[r_plot][c_plot]
            growth_data_curr = all_growth_data_np[r_plot][c_plot]
            apop_data_curr = all_apop_data_np[r_plot][c_plot]
            state_data_curr = all_state_data_np[r_plot][c_plot]
            p75_data_curr = all_p75_data_np[r_plot][c_plot]
            TrkB_data_curr = all_TrkB_data_np[r_plot][c_plot]
            p75_pro_data_curr = all_p75_pro_data_np[r_plot][c_plot]
            p75_B_data_curr = all_p75_B_data_np[r_plot][c_plot]
            TrkB_B_data_curr = all_TrkB_B_data_np[r_plot][c_plot]
            TrkB_pro_data_curr = all_TrkB_pro_data_np[r_plot][c_plot]
            tPA_data_curr = all_tPA_data_np[r_plot][c_plot]


            # Subplot 1: Pro/BDNF Dynamics & Activity Input
            axes[0].plot(times, P_data_curr, label='proBDNF (P)')
            axes[0].plot(times, B_data_curr, label='BDNF (B)')
            axes[0].plot(times, activity_data_curr, label='Activity Level (Input)', linestyle='--')
            axes[0].set_ylabel('Conc. (M)')
            axes[0].set_title('Pro/BDNF Dynamics & Activity Input')
            axes[0].legend(loc='upper right')
            axes[0].grid(True)

            # Subplot 2: Growth and Apoptosis Signals
            axes[1].plot(times, growth_data_curr, label='Growth Strength (TrkB signals)', color='green')
            axes[1].plot(times, apop_data_curr, label='Apoptosis Strength (p75 signals)', color='red')
            axes[1].set_ylabel('Signal (0-1)')
            axes[1].set_title('Growth/Apoptosis Signals')
            axes[1].legend(loc='upper right')
            axes[1].grid(True)

            # Subplot 3: Normalized Neuron State
            axes[2].plot(times, state_data_curr, label='Normalized Neuron State', color='purple')
            axes[2].axhline(network.threshold_growth, color='gray', linestyle=':', label=f'Growth Threshold ({network.threshold_growth})')
            axes[2].axhline(network.threshold_apoptosis, color='gray', linestyle=':', label=f'Apoptosis Threshold ({network.threshold_apoptosis})')
            axes[2].set_ylabel('State (-1 to 1)')
            axes[2].set_title('Normalized State')
            axes[2].legend(loc='upper right')
            axes[2].grid(True)

            # Subplot 4: Free Receptor Concentrations (p75 & TrkB)
            axes[3].plot(times, p75_data_curr, label='p75 Receptor')
            axes[3].plot(times, TrkB_data_curr, label='TrkB Receptor')
            axes[3].set_ylabel('Conc. (M)')
            axes[3].set_title('Free Receptor Concentrations')
            axes[3].legend(loc='upper right')
            axes[3].grid(True)

            # Subplot 5: Receptor-Ligand Complex Concentrations & tPA
            axes[4].plot(times, p75_pro_data_curr, label='p75-proBDNF Complex')
            axes[4].plot(times, p75_B_data_curr, label='p75-BDNF Complex')
            axes[4].plot(times, TrkB_B_data_curr, label='TrkB-BDNF Complex')
            axes[4].plot(times, TrkB_pro_data_curr, label='TrkB-proBDNF Complex')
            axes[4].plot(times, tPA_data_curr, label='tPA Enzyme', linestyle=':')
            axes[4].set_xlabel('Time (seconds)')
            axes[4].set_ylabel('Conc. (M)')
            axes[4].set_title('Complexes and tPA Dynamics')
            axes[4].legend(loc='upper right')
            axes[4].grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent overlapping, make room for suptitle
            plt.show(block=False) # Use block=False to open all figures without waiting

    # --- Separate Synaptic Weight Modulation Plot (Single Figure) ---
    if len(syn_weights_data) > 0:
        plt.figure(figsize=(10, 5)) # Separate figure for synapse weight
        syn_label = f"Weight: {target_synapse_for_plotting.name}" if target_synapse_for_plotting else "Synaptic Weight"
        plt.plot(times, syn_weights_data, label=syn_label, color='blue')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Weight')
        plt.title('Synaptic Weight Modulation (Selected Connection)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=True) # Use block=True for the last figure to keep windows open

    else:
        print("Skipping synaptic weight plot: No synaptic weight data recorded or no connections found.")


    # --- 12. Final Output Statements ---
    print("\n--- Final Simulation Results ---")
    for r_final in range(network.rows):
        for c_final in range(network.cols):
            neuron_obj_final = network.neurons[r_final][c_final]
            print(f"\n--- For Neuron: {neuron_obj_final.name} ---")
            final_conc = neuron_obj_final.get_concentrations()
            for key, val in final_conc.items():
                print(f"{key}: {val:.4e} M")
            print(f"Final Normalized Neuron State: {neuron_obj_final.calculate_and_get_neuron_state():.4f}")

    if target_synapse_for_plotting:
        print(f"\nFinal synaptic weight for {target_synapse_for_plotting.name}: {target_synapse_for_plotting.netcons[0].hoc.weight[0]:.4f}")
    else:
        print("\nNo specific synapse weight to report final value for.")

    print("\n--- Script Finished ---")