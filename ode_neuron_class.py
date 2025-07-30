

from neuron import h, nrn
from neuronpp.cells.cell import Cell
import numpy as np
import os



class ODENeuron(Cell):
    def __init__(self, name, initial_concentrations: list, params: list):
        super().__init__(name=name)
        
        self.soma = self.add_sec(name='soma', l=20, diam=20)

        self.insert('ode_neuron')

        self.soma_segment = self.soma(0.5)

        self.ode_mech = self.soma_segment.get_mechanism(name='ode_neuron')

        # Use h.Vector for recording, not h.PtrVector
        self.P = h.Vector()
        self.B = h.Vector()
        self.p75 = h.Vector()
        self.TrkB = h.Vector()
        self.p75_pro = h.Vector()
        self.p75_B = h.Vector()
        self.TrkB_B = h.Vector()
        self.TrkB_pro = h.Vector()
        self.tPA = h.Vector()
        self.activity_level_ref = h.Vector() # This is a STATE variable
        self.v_ref = h.Vector() # For membrane potential
        self.growth_strength_ref = h.Vector()
        self.apop_strength_ref = h.Vector()
        self.syn_input_activity_ref = h.Vector() # NEW: For reading syn_input_activity

        # Record the NMODL mechanism variables
        self.P.record(self.ode_mech._ref_P)
        self.B.record(self.ode_mech._ref_B)
        self.p75.record(self.ode_mech._ref_p75)
        self.TrkB.record(self.ode_mech._ref_TrkB)
        self.p75_pro.record(self.ode_mech._ref_p75_pro)
        self.p75_B.record(self.ode_mech._ref_p75_B)
        self.TrkB_B.record(self.ode_mech._ref_TrkB_B)
        self.TrkB_pro.record(self.ode_mech._ref_TrkB_pro)
        self.tPA.record(self.ode_mech._ref_tPA)
        self.activity_level_ref.record(self.ode_mech._ref_activity_level)
        self.v_ref.record(self.soma_segment.hoc._ref_v) # Membrane potential is on the segment
        self.growth_strength_ref.record(self.ode_mech._ref_growth_strength)
        self.apop_strength_ref.record(self.ode_mech._ref_apop_strength)
        self.syn_input_activity_ref.record(self.ode_mech._ref_syn_input_activity) # NEW RECORDING

        # Initialize NMODL STATE variables with initial_concentrations
        self.ode_mech.P = initial_concentrations[0]
        self.ode_mech.B = initial_concentrations[1]
        self.ode_mech.p75 = initial_concentrations[2]
        self.ode_mech.TrkB = initial_concentrations[3]
        self.ode_mech.p75_pro = initial_concentrations[4]
        self.ode_mech.p75_B = initial_concentrations[5]
        self.ode_mech.TrkB_B = initial_concentrations[6]
        self.ode_mech.TrkB_pro = initial_concentrations[7]
        self.ode_mech.tPA = initial_concentrations[8]
        self.ode_mech.activity_level = 0.0 # Explicitly initialize
        self.ode_mech.syn_input_activity = 0.0 # NEW: Explicitly initialize

        # Set NMODL PARAMETER variables using the params list
        param_names = [
            "ksP", "k_cleave", "k_p75_pro_on", "k_p75_pro_off", "k_degP", "k_TrkB_pro_on", "k_TrkB_pro_off",
            "k_TrkB_B_on", "k_TrkB_B_off", "k_degB", "k_p75_B_on", "k_p75_B_off", "k_degR1", "k_degR2",
            "k_int_p75_pro", "k_int_p75_B", "k_int_TrkB_B", "k_int_TrkB_pro", "aff_p75_pro",
            "aff_p75_B", "aff_TrkB_pro", "aff_TrkB_B", "k_deg_tPA", "ks_tPA", "ks_p75", "ks_TrkB",
            "tau_activity", "activity_gain", "g_leak", "e_leak", "v_threshold_spike"
        ]

        for i, param_name in enumerate(param_names):
            if i < len(params):
                setattr(self.ode_mech, param_name, params[i])
            else:
                print(f"Warning: Parameter '{param_name}' not found in provided params list for {self.name}. Using NMODL default.")

        # Spike detector: for recording spikes (optional)
        # This is just for spike recording, not for network connections
        self.spike_detector = h.NetCon(self.soma_segment.hoc._ref_v, None, sec=self.soma.hoc)
        self.spike_detector.threshold = self.ode_mech.v_threshold_spike
        self.spike_detector.delay = 0.0 # Detect instantaneously
        
        # Vector to record spike times
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)

        self.neuron_state = 0.0
        self.stim = None # For IClamp

        print(f"ODENeuron '{self.name}' initialized.")

    def add_external_current_stim(self, delay: float, dur: float, amp: float):
        self.stim = h.IClamp(self.soma_segment.hoc)
        self.stim.delay = delay
        self.stim.dur = dur
        self.stim.amp = amp

    def update_activity_level(self, activity_value: float):
        """Method to update the activity level - you'll need to implement this"""
        # This method wasn't in your original code but is called in the main loop
        # You might need to implement this based on your NMODL mechanism
        self.ode_mech.syn_input_activity = activity_value

    def calculate_and_get_neuron_state(self) -> float:
        growth_strength = self.ode_mech.growth_strength
        apop_strength = self.ode_mech.apop_strength
        signal = growth_strength - apop_strength
        self.neuron_state = signal
        return self.neuron_state
    
    def get_concentrations(self) -> dict:
        return {
            'P': self.ode_mech.P,
            'B': self.ode_mech.B,
            'p75': self.ode_mech.p75,
            'TrkB': self.ode_mech.TrkB,
            'p75_pro': self.ode_mech.p75_pro,
            'p75_B': self.ode_mech.p75_B,
            'TrkB_B': self.ode_mech.TrkB_B,
            'TrkB_pro': self.ode_mech.TrkB_pro,
            'tPA': self.ode_mech.tPA,
            'activity_level': self.ode_mech.activity_level,
            'syn_input_activity': self.ode_mech.syn_input_activity, # NEW: Include this
            'v': self.soma_segment.hoc.v
        }
    
    def get_params(self) -> dict:
        param_names = [
            "ksP", "k_cleave", "k_p75_pro_on", "k_p75_pro_off", "k_degP", "k_TrkB_pro_on", "k_TrkB_pro_off",
            "k_TrkB_B_on", "k_TrkB_B_off", "k_degB", "k_p75_B_on", "k_p75_B_off", "k_degR1", "k_degR2",
            "k_int_p75_pro", "k_int_p75_B", "k_int_TrkB_B", "k_int_TrkB_pro", "aff_p75_pro",
            "aff_p75_B", "aff_TrkB_pro", "aff_TrkB_B", "k_deg_tPA", "ks_tPA", "ks_p75", "ks_TrkB",
            "tau_activity", "activity_gain", "g_leak", "e_leak", "v_threshold_spike"
        ]
        
        param_dict = {}
        for param_name in param_names:
            param_dict[param_name] = getattr(self.ode_mech, param_name)
        return param_dict

# The __main__ block for ODENeuron testing is now superseded by the network script
# but can be kept for standalone testing of ODENeuron if desired.
    
if __name__ == "__main__":
    # Ensure NEURON's working directory is set correctly for mod file compilation
    # This automatically compiles .mod files in the current directory or 'mods' subdirectory
    # If your .mod file is in a 'mods' folder relative to this script:
    # nrn.mod_func('initmod') # Not always necessary if neuronpp handles it, but good practice if issues arise.
    
    # NEURON's standard way to load mechanisms:
    # Check if 'mods' directory exists
    mod_dir = os.path.join(os.path.dirname(__file__), 'mods')
    if os.path.exists(mod_dir):
        print(f"Compiling NMODL files in {mod_dir}...")
        os.system(f"nrnivmodl {mod_dir}")
    else:
        print("Warning: 'mods' directory not found. Assuming ode_neuron.mod is in the current directory.")
        print("Attempting to compile NMODL files in current directory...")
        os.system("nrnivmodl") # Compile mods in current directory

    # Import the compiled mechanism after compilation
    h.nrn_load_dll(os.path.join("arm64", ".libs", "libnrnmech.dylib")) # Adjust for your OS (e.g., .dll for Windows)

    # Initial concentrations (y0) from your original ODE code
    initial_concentrations = [
        0.2,   # P: proBDNF
        0.0,   # B: BDNF
        1.0,   # p75: free p75 receptor
        1.0,   # TrkB: free TrkB receptor
        0.0,   # p75_pro: proBDNF-p75 complex
        0.0,   # p75_B: BDNF-p75 complex
        0.0,   # TrkB_B: BDNF-TrkB complex
        0.0,   # TrkB_pro: proBDNF-TrkB complex
        0.1    # tPA: tPA enzyme
    ]

    # Parameters (params) from your original ODE code
    # Ensure this order matches the NMODL PARAMETER block
    parameters = [
        5.0e-3,   # ksP
        0.01,    # k_cleave
        1.0,    # k_p75_pro_on
        0.9,    # k_p75_pro_off
        5.0e-4,   # k_degP
        0.2,    # k_TrkB_pro_on
        0.1,   # k_TrkB_pro_off
        1.0,    # k_TrkB_B_on
        0.9,    # k_TrkB_B_off
        0.15,    # k_degB
        0.3,    # k_p75_B_on
        0.2,   # k_p75_B_off
        0.0001,   # k_degR1
        0.00001,   # k_degR2
        0.0005,    # k_int_p75_pro
        0.0005,    # k_int_p75_B
        0.0005,    # k_int_TrkB_B
        0.0005,    # k_int_TrkB_pro
        0.9,    # aff_p75_pro
        0.1,    # aff_p75_B
        0.1,    # aff_TrkB_pro
        0.9,    # aff_TrkB_B
        0.011,   # k_deg_tPA
        0.0001,    # ks_tPA
        0.0001,    # ks_p75
        0.00001    # ks_TrkB
    ]
    from neuron import h
# Print the names of all density mechanisms
    mt = h.MechanismType(0)
    mname  = h.ref('')
    for i in range(mt.count()):
        mt.select(i)
        mt.selected(mname)
        print(mname[0])
    # Initialize a CustomNeuron
    neuron1 = ODENeuron(name="neuron1",
                           initial_concentrations=initial_concentrations,
                           params=parameters)
    

    # Setup NEURON simulation (standard for NEURON)
    h.finitialize(-65) # Initialize membrane potential (not directly relevant for ODEs, but good practice)
    h.t = 0
    h.dt = 0.01 # seconds (10 ms) - keeping your adjusted dt

    # Initialize standard Python lists to store data
    t_list = []
    P_list = []
    B_list = []
    growth_list = []
    apop_list = []
    state_list = []
    activity_list = []
    # --- ADD THESE NEW LISTS ---
    p75_list = []
    TrkB_list = []
    p75_pro_list = []
    p75_B_list = []
    TrkB_B_list = []
    TrkB_pro_list = []
    tPA_list = []
    # --- END NEW LISTS ---

    # Simulation loop
    runtime = 10000 # seconds
    num_steps = int(runtime / h.dt)

    # Introduce some activity changes during simulation
    burst_duration_sec = 500  # seconds
    inter_burst_interval_sec = 1000 # seconds
    high_activity_value = 110
    low_activity_value = 0.01

    for i in range(num_steps):
        current_time = h.t
        time_within_cycle = current_time % inter_burst_interval_sec
        if time_within_cycle < burst_duration_sec:
            current_activity = high_activity_value
        else:
            current_activity = low_activity_value

        neuron1.update_activity_level(current_activity)

        h.fadvance() # Advance the simulation by h.dt

        # Manually append the numerical values at each time step
        t_list.append(h.t)
        P_list.append(neuron1.P[0])
        B_list.append(neuron1.B[0])
        growth_list.append(neuron1.growth_strength_ref[0])
        apop_list.append(neuron1.apop_strength_ref[0])
        activity_list.append(neuron1.activity_level_ref[0])
        state_list.append(neuron1.calculate_and_get_neuron_state()) # This was already here

        # --- ADD THESE NEW APPENDS ---
        p75_list.append(neuron1.p75[0])
        TrkB_list.append(neuron1.TrkB[0])
        p75_pro_list.append(neuron1.p75_pro[0])
        p75_B_list.append(neuron1.p75_B[0])
        TrkB_B_list.append(neuron1.TrkB_B[0])
        TrkB_pro_list.append(neuron1.TrkB_pro[0])
        tPA_list.append(neuron1.tPA[0])
        # --- END NEW APPENDS ---

    # Convert lists to numpy arrays for plotting
    times = np.array(t_list)
    P_data = np.array(P_list)
    B_data = np.array(B_list)
    growth_data = np.array(growth_list)
    apop_data = np.array(apop_list)
    state_data = np.array(state_list)
    activity_data = np.array(activity_list)
    # --- ADD THESE NEW CONVERSIONS ---
    p75_data = np.array(p75_list)
    TrkB_data = np.array(TrkB_list)
    p75_pro_data = np.array(p75_pro_list)
    p75_B_data = np.array(p75_B_list)
    TrkB_B_data = np.array(TrkB_B_list)
    TrkB_pro_data = np.array(TrkB_pro_list)
    tPA_data = np.array(tPA_list)
    # --- END NEW CONVERSIONS ---

    # Plotting results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 12)) # Increased figure size for more subplots

    # --- Subplot 1: Pro/BDNF Dynamics & Activity Input (Existing) ---
    plt.subplot(5, 1, 1) # Changed to 5 rows for 5 plots
    plt.plot(times, P_data, label='proBDNF (P)')
    plt.plot(times, B_data, label='BDNF (B)')
    plt.plot(times, activity_data, label='Activity Level (Input)', linestyle='--')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Concentration (uM)')
    plt.title(f'Neuron: {neuron1.name} - Pro/BDNF Dynamics & Activity Input')
    plt.legend()
    plt.grid(True)

    # --- Subplot 2: Growth and Apoptosis Signals (Existing) ---
    plt.subplot(5, 1, 2) # Changed to 5 rows
    plt.plot(times, growth_data, label='Growth Strength (TrkB signals)', color='green')
    plt.plot(times, apop_data, label='Apoptosis Strength (p75 signals)', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Strength (0-1)')
    plt.title(f'Neuron: {neuron1.name} - Growth and Apoptosis Signals')
    plt.legend()
    plt.grid(True)

    # --- Subplot 3: Normalized Neuron State (Existing) ---
    plt.subplot(5, 1, 3) # Changed to 5 rows
    plt.plot(times, state_data, label='Normalized Neuron State', color='purple')
    plt.axhline(0.2, color='gray', linestyle=':', label='Growth Threshold (+0.2)')
    plt.axhline(-0.2, color='gray', linestyle=':', label='Apoptosis Threshold (-0.2)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized State (-1 to 1)')
    plt.title(f'Neuron: {neuron1.name} - Combined Growth/Apoptosis State')
    plt.legend()
    plt.grid(True)

    # --- NEW Subplot 4: Free Receptor Concentrations (p75 & TrkB) ---
    plt.subplot(5, 1, 4) # New subplot
    # You'll need to add p75_list and TrkB_list to your recording section above the plotting.
    p75_list = []
    TrkB_list = []
    # Add these to your loop:
    # p75_list.append(neuron1.p75[0])
    # TrkB_list.append(neuron1.TrkB[0])
    # ... and then convert to numpy arrays:
    # p75_data = np.array(p75_list)
    # TrkB_data = np.array(TrkB_list)

    plt.plot(times, p75_data, label='p75 Receptor')
    plt.plot(times, TrkB_data, label='TrkB Receptor')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Concentration (uM)')
    plt.title('Free Receptor Concentrations')
    plt.legend()
    plt.grid(True)

    # --- NEW Subplot 5: Receptor-Ligand Complex Concentrations & tPA ---
    plt.subplot(5, 1, 5) # New subplot
    # You'll need to add lists for these to your recording section and convert to numpy arrays.
    # p75_pro_list = []
    # p75_B_list = []
    # TrkB_B_list = []
    # TrkB_pro_list = []
    # tPA_list = []
    # Add these to your loop:
    # p75_pro_list.append(neuron1.p75_pro[0])
    # p75_B_list.append(neuron1.p75_B[0])
    # TrkB_B_list.append(neuron1.TrkB_B[0])
    # TrkB_pro_list.append(neuron1.TrkB_pro[0])
    # tPA_list.append(neuron1.tPA[0])
    # ... and then convert to numpy arrays:
    # p75_pro_data = np.array(p75_pro_list)
    # p75_B_data = np.array(p75_B_list)
    # TrkB_B_data = np.array(TrkB_B_list)
    # TrkB_pro_data = np.array(TrkB_pro_list)
    # tPA_data = np.array(tPA_list)

    plt.plot(times, p75_pro_data, label='p75-proBDNF Complex')
    plt.plot(times, p75_B_data, label='p75-BDNF Complex')
    plt.plot(times, TrkB_B_data, label='TrkB-BDNF Complex')
    plt.plot(times, TrkB_pro_data, label='TrkB-proBDNF Complex')
    plt.plot(times, tPA_data, label='tPA Enzyme', linestyle=':')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Concentration (uM)')
    plt.title('Receptor-Ligand Complexes and tPA Dynamics')
    plt.legend()
    plt.grid(True)


    plt.tight_layout() # Adjust layout to prevent overlapping
    plt.show()

    print("\nFinal concentrations:")
    final_conc = neuron1.get_concentrations()
    for key, val in final_conc.items():
        print(f"{key}: {val:.4e} uM") # Changed uM to M to be consistent with plots if units are M
    print(f"Final Normalized Neuron State: {neuron1.calculate_and_get_neuron_state():.4f}")
