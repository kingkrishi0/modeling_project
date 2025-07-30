

from neuron import h, nrn
from neuronpp.cells.cell import Cell
from neuronpp.core.hocwrappers.synapses.single_synapse import SingleSynapse
import numpy as np
from neuronpp.core.hocwrappers.point_process import PointProcess
import os
from ode_neuron_class import ODENeuron

from neuron import h, nrn
from neuronpp.cells.cell import Cell
from neuronpp.core.hocwrappers.synapses.single_synapse import SingleSynapse
import numpy as np
from neuronpp.core.hocwrappers.point_process import PointProcess
import os

class NeuronalNetwork:
    def __init__(self, rows: int, cols: int, initial_neuron_concentrations: list, neuron_params: list,
                 synapse_type: str = "ProbabilisticSyn", # Use ProbabilisticSyn
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
                neuron = ODENeuron(name=f'neuron_r{r}c{c}', initial_concentrations=initial_neuron_concentrations, 
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
        
        post_segment_loc = post_neuron.soma(0.5) 
        h_section_at_segment = (post_segment_loc.parent.hoc)(post_segment_loc.x)

        if hasattr(h, self.synapse_type):
            raw_h_point_process = getattr(h, self.synapse_type)(h_section_at_segment)
        else:
            raise ValueError(f"Synapse type '{self.synapse_type}' is not a recognized NEURON PointProcess. "
                             f"Ensure it's compiled and correct. (e.g., ProbabilisticSyn)")

        point_process_mech = PointProcess(
            hoc_obj=raw_h_point_process,
            parent=post_segment_loc.parent,
            name=f"{self.synapse_type}_{conn_id}",
            cell = post_neuron,
            mod_name = self.synapse_type
        )
        
        # Primary NetCon: from presynaptic neuron's spike detector to the postsynaptic ProbabilisticSyn
        netcon_to_synapse = h.NetCon(pre_neuron.spike_detector, point_process_mech.hoc, sec=pre_neuron.soma.hoc)
        netcon_to_synapse.weight[0] = self.initial_syn_weight
        netcon_to_synapse.delay = 1.0 # Standard synaptic delay

        # LINK THE POINTER: This is crucial for ProbabilisticSyn to write to ODENeuron's syn_input_activity
        # The POINTER declaration in ProbabilisticSyn (target_syn_input_activity)
        # needs to point to the address of the syn_input_activity variable in the ODENeuron mechanism.
        point_process_mech.hoc.target_syn_input_activity = post_neuron.ode_mech._ref_syn_input_activity
        
        syn_obj = SingleSynapse(
            source=pre_neuron.spike_detector, # Source for neuronpp wrapper, though netcon handles actual connection
            point_process=point_process_mech, 
            name=conn_id 
        )
        # Add the actual NetCon to the SingleSynapse's list for later access/modulation
        syn_obj.netcons.append(netcon_to_synapse)


        self.connections.append({
            'pre_neuron': pre_neuron,
            'post_neuron': post_neuron,
            'syn': syn_obj, # This is the SingleSynapse object
            'netcon_to_synapse': netcon_to_synapse # Store the main NetCon for weight modulation
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
                offsets = [(0,1), (1, 0)] # Connect right and down (unidirectional first)

                for dr, dc in offsets:
                    post_r, post_c = r+dr, c+dc

                    post_neuron = self._get_neuron_at_grid_pos(post_r, post_c)

                    if post_neuron:
                        self._create_connection(pre_neuron, post_neuron)
                        self._create_connection(post_neuron, pre_neuron) # Bidirectional for all

    def modulate_synaptic_weights(self):
        for conn_info in self.connections:
            pre_neuron = conn_info['pre_neuron']
            post_neuron = conn_info['post_neuron']
            netcon_to_synapse = conn_info['netcon_to_synapse']

            pre_n_state = pre_neuron.calculate_and_get_neuron_state()
            post_n_state = post_neuron.calculate_and_get_neuron_state()

            current_weight = netcon_to_synapse.weight[0]
            new_weight = current_weight

            if pre_n_state > self.threshold_growth and post_n_state > self.threshold_growth:
                new_weight += self.learning_rate


            elif pre_n_state < self.threshold_apoptosis or post_n_state < self.threshold_apoptosis:
                new_weight -= self.learning_rate

            new_weight = max(self.min_weight, new_weight)
            new_weight = min(self.max_weight, new_weight)

            netcon_to_synapse.weight[0] = new_weight

# Assuming ODENeuron and NeuronalNetwork classes are defined above this point
# and neuronpp imports are correct.
# Ensure mods/ode_neuron.mod and mods/probabilistic_syn.mod are present.

if __name__ == "__main__":
    print("--- Starting Simulation Script ---")

    # --- 1. Compile NMODL files ---
    mod_dir = os.path.join(os.path.dirname(__file__), 'mods')
    if os.path.exists(mod_dir):
        print(f"Compiling NMODL files in {mod_dir}...")
        os.system(f"nrnivmodl {mod_dir}")
    else:
        print("Warning: 'mods' directory not found. Attempting to compile NMODL files in current directory...")
        os.system("nrnivmodl") # Compile mods in current directory

    try:
        # Adjust path for your OS (e.g., .dll for Windows, .dylib for macOS)
        h.nrn_load_dll(os.path.join("x86_64", ".libs", "libnrnmech.so"))
    except Exception as e:
        print(f"Error loading NEURON mechanisms: {e}")
        print("Please ensure NMODL files are compiled and 'libnrnmech.so' (or .dll/.dylib) is in the correct path.")
        exit()

    # --- 2. Define Initial Concentrations and Parameters ---
    initial_neuron_concentrations = [
        0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1
    ]
    # neuron_parameters list order MUST match the `param_names` list in ODENeuron.__init__
    # Make sure you have enough values here for all 32 parameters including the new ones
    neuron_parameters = [
        5.0e-3, 0.01, 1.0, 0.9, 5.0e-4, 0.2, 0.1, 1.0, 0.9, 0.005, 0.3, 0.1, 0.0001, 0.00001,
        0.0005, 0.0005, 0.0005, 0.0005, 0.9, 0.1, 0.1, 0.9, 0.0011, 0.0001, 0.0001, 0.00001, # Your original 26 params
        50.0,    # tau_activity (ms) - index 26
        0.1,     # activity_gain (unitless) - index 27
        1.0,     # cm (uF/cm2) - index 28
        0.0001,  # g_leak (S/cm2) - index 29
        -65.0,   # e_leak (mV) - index 30
        -20.0    # v_threshold_spike (mV) - index 31
    ]

    # --- 3. Create the Neuronal Network Instance ---
    grid_rows = 2
    grid_cols = 2
    network = NeuronalNetwork(rows=grid_rows, cols=grid_cols,
                              initial_neuron_concentrations=initial_neuron_concentrations,
                              neuron_params=neuron_parameters,
                              synapse_type="ProbabilisticSyn", # Use the new ProbabilisticSyn
                              initial_syn_weight=0.5,
                              learning_rate=0.0005,
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
        print(f"Initial ProBDNF (P) of {first_neuron.name}: {first_neuron.get_concentrations()['P']:.4e} uM")

    if len(network.connections) > 0:
        first_connection_info = network.connections[0]
        pre_n = first_connection_info['pre_neuron']
        post_n = first_connection_info['post_neuron']
        syn = first_connection_info['syn'] # SingleSynapse object
        netcon_main = first_connection_info['netcon_to_synapse'] # The NetCon we modulate

        print(f"\nExample Connection: {pre_n.name} -> {post_n.name}")
        print(f"  Initial Weight: {netcon_main.weight[0]:.4f}")
        print(f"  Synapse Type: {syn.point_process_name}")
        print(f"  Synapse Delay: {netcon_main.delay:.4f}")

        try:
            print(f"  Pre-synaptic Source for NetCon: {netcon_main.source.sec.name()}({netcon_main.source.x:.2f})")
            print(f"  Post-synaptic Target for NetCon: {netcon_main.target.sec.name()}({netcon_main.target.x:.2f})")
            
            # Verify the POINTER connection (from ProbabilisticSyn to ODENeuron's syn_input_activity)
            # This is accessed directly through the point_process_mech's hoc object
            # and its target_syn_input_activity pointer.
            print(f"  Synaptic Activity Pointer Set: {syn.point_process.hoc.target_syn_input_activity is not None}")
            if syn.point_process.hoc.target_syn_input_activity is not None:
                print(f"  Pointer target: {syn.point_process.hoc.target_syn_input_activity.name() if hasattr(syn.point_process.hoc.target_syn_input_activity, 'name') else 'NMODL variable'}")

        except AttributeError:
            print("  Could not retrieve information for pointer (unexpected structure or not set).")
    else:
        print("\nNo connections created in the network.")

    # --- 5. Setup NEURON simulation parameters ---
    h.finitialize(-65) # Initialize membrane potential
    h.t = 0
    h.dt = 0.025 # seconds (25 ms) - Adjusted for better balance with ODEs.

    # --- 6. Recording Variables (for Plotting) ---
    t_list = []
    all_P_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_B_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_growth_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_apop_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_state_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_activity_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)] # activity_level_ref
    all_p75_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_TrkB_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_p75_pro_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_p75_B_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_TrkB_B_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_TrkB_pro_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_tPA_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)]
    all_v_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)] # Membrane potential
    all_syn_input_activity_data = [[[] for _ in range(network.cols)] for _ in range(network.rows)] # NEW: syn_input_activity

    # Recording for a specific synapse's weight (e.g., neuron_r0c0 -> neuron_r0c1)
    target_synapse_for_plotting = None
    if grid_cols > 1:
        pre_n_r0c0 = network.neurons[0][0]
        post_n_r0c1 = network.neurons[0][1]
        for conn in network.connections:
            if conn['pre_neuron'] == pre_n_r0c0 and conn['post_neuron'] == post_n_r0c1:
                target_synapse_for_plotting = conn
                break
    if not target_synapse_for_plotting and len(network.connections) > 0:
        target_synapse_for_plotting = network.connections[0]
        print("Note: Plotting the weight of the first available connection.")
    elif not target_synapse_for_plotting:
        print("Note: No synapses found to plot weights.")

    syn_weights_list = []

    # --- Inject current into neuron_r0c0 to make it spike ---
    network.neurons[0][0].add_external_current_stim(delay=50, dur=1500, amp=0.5)

    # --- Simulation loop ---
    runtime = 2000 # seconds
    num_steps = int(runtime / h.dt)

    print("\n--- Starting Simulation Loop ---")
    for i in range(num_steps):
        # Record data for all neurons
        for r_idx in range(network.rows):
            for c_idx in range(network.cols):
                neuron_obj = network.neurons[r_idx][c_idx]
                
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
                all_v_data[r_idx][c_idx].append(neuron_obj.v_ref[0])
                all_syn_input_activity_data[r_idx][c_idx].append(neuron_obj.syn_input_activity_ref[0]) # NEW RECORDING

        h.fadvance() # Advance simulation

        t_list.append(h.t)

        if target_synapse_for_plotting:
            syn_weights_list.append(target_synapse_for_plotting['netcon_to_synapse'].weight[0])

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
    all_v_data_np = [[np.array(all_v_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]
    all_syn_input_activity_data_np = [[np.array(all_syn_input_activity_data[r][c]) for c in range(network.cols)] for r in range(network.rows)]

    syn_weights_data = np.array(syn_weights_list)

    # --- 10. Plotting Results ---
    import matplotlib.pyplot as plt

    # Plot membrane potential for each neuron
    for r_plot in range(network.rows):
        for c_plot in range(network.cols):
            neuron_name = network.neurons[r_plot][c_plot].name
            plt.figure(figsize=(12, 4))
            plt.plot(times, all_v_data_np[r_plot][c_plot], label=f'{neuron_name} Membrane Potential (Vm)')
            plt.xlabel('Time (s)')
            plt.ylabel('Vm (mV)')
            plt.title(f'Membrane Potential of {neuron_name}')
            plt.legend()
            plt.grid(True)
            plt.show(block=False)

    for r_plot in range(network.rows):
        for c_plot in range(network.cols):
            neuron_name = network.neurons[r_plot][c_plot].name
            
            fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True) # Changed to 6 subplots
            fig.suptitle(f'Dynamics for Neuron: {neuron_name}', fontsize=16)

            P_data_curr = all_P_data_np[r_plot][c_plot]
            B_data_curr = all_B_data_np[r_plot][c_plot]
            syn_activity_level_data_curr = all_activity_data_np[r_plot][c_plot] # Activity_level STATE
            syn_input_activity_data_curr = all_syn_input_activity_data_np[r_plot][c_plot] # Raw input from synapses
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


            # Subplot 1: Pro/BDNF Dynamics
            axes[0].plot(times, P_data_curr, label='proBDNF (P)')
            axes[0].plot(times, B_data_curr, label='BDNF (B)')
            axes[0].set_ylabel('Conc. (uM)')
            axes[0].set_title('Pro/BDNF Dynamics')
            axes[0].legend(loc='upper right')
            axes[0].grid(True)
            
            # Subplot 2: Synaptic Activity Input & Activity Level State
            axes[1].plot(times, syn_input_activity_data_curr, label='Raw Synaptic Input Activity', linestyle=':', color='orange')
            axes[1].plot(times, syn_activity_level_data_curr, label='Activity Level (ODE State)', color='blue')
            axes[1].set_ylabel('Activity (unitless)')
            axes[1].set_title('Synaptic Activity Input & ODE Activity Level')
            axes[1].legend(loc='upper right')
            axes[1].grid(True)

            # Subplot 3: Growth and Apoptosis Signals
            axes[2].plot(times, growth_data_curr, label='Growth Strength (TrkB signals)', color='green')
            axes[2].plot(times, apop_data_curr, label='Apoptosis Strength (p75 signals)', color='red')
            axes[2].set_ylabel('Signal (0-1)')
            axes[2].set_title('Growth/Apoptosis Signals')
            axes[2].legend(loc='upper right')
            axes[2].grid(True)

            # Subplot 4: Normalized Neuron State
            axes[3].plot(times, state_data_curr, label='Normalized Neuron State', color='purple')
            axes[3].axhline(network.threshold_growth, color='gray', linestyle=':', label=f'Growth Threshold ({network.threshold_growth})')
            axes[3].axhline(network.threshold_apoptosis, color='gray', linestyle=':', label=f'Apoptosis Threshold ({network.threshold_apoptosis})')
            axes[3].set_ylabel('State (-1 to 1)')
            axes[3].set_title('Normalized State')
            axes[3].legend(loc='upper right')
            axes[3].grid(True)

            # Subplot 5: Free Receptor Concentrations (p75 & TrkB)
            axes[4].plot(times, p75_data_curr, label='p75 Receptor')
            axes[4].plot(times, TrkB_data_curr, label='TrkB Receptor')
            axes[4].set_ylabel('Conc. (uM)')
            axes[4].set_title('Free Receptor Concentrations')
            axes[4].legend(loc='upper right')
            axes[4].grid(True)

            # Subplot 6: Receptor-Ligand Complex Concentrations & tPA
            axes[5].plot(times, p75_pro_data_curr, label='p75-proBDNF Complex')
            axes[5].plot(times, p75_B_data_curr, label='p75-BDNF Complex')
            axes[5].plot(times, TrkB_B_data_curr, label='TrkB-BDNF Complex')
            axes[5].plot(times, TrkB_pro_data_curr, label='TrkB-proBDNF Complex')
            axes[5].plot(times, tPA_data_curr, label='tPA Enzyme', linestyle=':')
            axes[5].set_xlabel('Time (seconds)')
            axes[5].set_ylabel('Conc. (uM)')
            axes[5].set_title('Complexes and tPA Dynamics')
            axes[5].legend(loc='upper right')
            axes[5].grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.show(block=False)

    if len(syn_weights_data) > 0:
        plt.figure(figsize=(10, 5))
        syn_label = f"Weight: {target_synapse_for_plotting['pre_neuron'].name} -> {target_synapse_for_plotting['post_neuron'].name}" if target_synapse_for_plotting else "Synaptic Weight"
        plt.plot(times, syn_weights_data, label=syn_label, color='blue')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Weight')
        plt.title('Synaptic Weight Modulation (Selected Connection)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=True)
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
                print(f"{key}: {val:.4e} uM")
            print(f"Final Normalized Neuron State: {neuron_obj_final.calculate_and_get_neuron_state():.4f}")

    if target_synapse_for_plotting:
        print(f"\nFinal synaptic weight for {target_synapse_for_plotting['pre_neuron'].name} -> {target_synapse_for_plotting['post_neuron'].name}: {target_synapse_for_plotting['netcon_to_synapse'].weight[0]:.4f}")
    else:
        print("\nNo specific synapse weight to report final value for.")

    print("\n--- Script Finished ---")