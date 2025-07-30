"""
Diagnose and fix the voltage explosion issue
The problem is likely in the membrane equation or leak conductance
"""

from neuron import h, nrn
import os
import numpy as np
import matplotlib.pyplot as plt
from ode_neuron_class import ODENeuron

def load_mechanisms():
    """Load NEURON mechanisms"""
    mod_dir = os.path.join(os.path.dirname(__file__), 'mods')
    if os.path.exists(mod_dir):
        os.system(f"nrnivmodl {mod_dir}")
    else:
        os.system("nrnivmodl")
    
    lib_paths = [
        os.path.join("x86_64", ".libs", "libnrnmech.so"),
        os.path.join("arm64", ".libs", "libnrnmech.dylib"),
        "nrnmech.dll"
    ]
    
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            try:
                h.nrn_load_dll(lib_path)
                print(f"Loaded: {lib_path}")
                return True
            except:
                continue
    return False

def test_leak_conductance_values():
    """Test different leak conductance values to find stable range"""
    print("=== Testing Different Leak Conductance Values ===")
    
    if not load_mechanisms():
        return
    
    # Base parameters
    initial_concentrations = [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    
    # Test different g_leak values
    g_leak_values = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    
    for g_leak in g_leak_values:
        print(f"\n--- Testing g_leak = {g_leak} S/cm² ---")
        
        # Parameters with varying g_leak
        neuron_parameters = [
            5.0e-3, 0.01, 1.0, 0.9, 5.0e-4, 0.2, 0.1, 1.0, 0.9, 0.005, 
            0.3, 0.1, 0.0001, 0.00001, 0.0005, 0.0005, 0.0005, 0.0005, 
            0.9, 0.1, 0.1, 0.9, 0.0011, 0.0001, 0.0001, 0.00001,
            50.0, 0.1, g_leak, -65.0, -20.0  # g_leak is index 28
        ]
        
        try:
            neuron = ODENeuron(f"test_neuron_gleak_{g_leak}", initial_concentrations, neuron_parameters)
            
            # Add moderate stimulation
            stim = h.IClamp(neuron.soma_segment.hoc)
            stim.delay = 10
            stim.dur = 50
            stim.amp = 5.0  # Moderate stimulation
            
            # Record voltage
            t_vec = h.Vector()
            v_vec = h.Vector()
            t_vec.record(h._ref_t)
            v_vec.record(neuron.soma_segment.hoc._ref_v)
            
            # Run short simulation
            h.finitialize(-65)
            h.dt = 0.025
            h.tstop = 100
            
            # Monitor for explosion
            max_v = -65
            min_v = -65
            exploded = False
            
            while h.t < h.tstop and not exploded:
                h.fadvance()
                current_v = neuron.soma_segment.hoc.v
                max_v = max(max_v, current_v)
                min_v = min(min_v, current_v)
                
                # Check for explosion
                if abs(current_v) > 1000:
                    exploded = True
                    print(f"❌ EXPLODED at t={h.t:.2f}ms, V={current_v:.1f}mV")
                    break
            
            if not exploded:
                times = np.array(t_vec)
                voltages = np.array(v_vec)
                print(f"✅ Stable: V range = {voltages.min():.1f} to {voltages.max():.1f} mV")
                
                # Check if it can spike properly
                if voltages.max() > 0:
                    print(f"   Can depolarize above 0 mV")
                if voltages.max() > 20:
                    print(f"   Can reach action potential levels")
                
                return g_leak  # Return first stable value
            
        except Exception as e:
            print(f"❌ Error with g_leak={g_leak}: {e}")
    
    return None

def test_with_stable_parameters():
    """Test with known stable parameters"""
    print("\n=== Testing with Conservative Stable Parameters ===")
    
    if not load_mechanisms():
        return
    
    # Use more conservative parameters
    initial_concentrations = [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    
    # CONSERVATIVE parameters - focus on stability
    neuron_parameters = [
        5.0e-3, 0.01, 1.0, 0.9, 5.0e-4, 0.2, 0.1, 1.0, 0.9, 0.005, 
        0.3, 0.1, 0.0001, 0.00001, 0.0005, 0.0005, 0.0005, 0.0005, 
        0.9, 0.1, 0.1, 0.9, 0.0011, 0.0001, 0.0001, 0.00001,
        50.0,      # tau_activity
        0.1,       # activity_gain  
        0.0003,    # g_leak - CONSERVATIVE VALUE
        -65.0,     # e_leak
        -20.0      # v_threshold_spike
    ]
    
    neuron = ODENeuron("stable_test", initial_concentrations, neuron_parameters)
    
    print(f"Parameters set:")
    print(f"  g_leak = {neuron.ode_mech.g_leak} S/cm²")
    print(f"  e_leak = {neuron.ode_mech.e_leak} mV")
    print(f"  v_threshold = {neuron.ode_mech.v_threshold_spike} mV")
    
    # Add graduated stimulation
    stim = h.IClamp(neuron.soma_segment.hoc)
    stim.delay = 50
    stim.dur = 100
    stim.amp = 8.0  # Strong but not excessive
    
    # Record everything
    t_vec = h.Vector()
    v_vec = h.Vector()
    i_vec = h.Vector()  # Record current
    
    t_vec.record(h._ref_t)
    v_vec.record(neuron.soma_segment.hoc._ref_v)
    i_vec.record(neuron.ode_mech._ref_i)  # Membrane current from ode_neuron
    
    # Record spikes
    spike_times = h.Vector()
    neuron.spike_detector.record(spike_times)
    
    # Run simulation with monitoring
    h.finitialize(-65)
    h.dt = 0.025
    h.tstop = 300
    
    print("Running stable parameter test...")
    
    step = 0
    while h.t < h.tstop:
        h.fadvance()
        
        current_v = neuron.soma_segment.hoc.v
        
        # Safety check
        if abs(current_v) > 200:
            print(f"⚠️  High voltage detected: {current_v:.1f}mV at t={h.t:.1f}ms")
            if abs(current_v) > 500:
                print("❌ Stopping due to voltage explosion")
                break
        
        step += 1
        if step % 2000 == 0:  # Every 50ms
            print(f"  t={h.t:.1f}ms, V={current_v:.2f}mV")
    
    # Process results
    times = np.array(t_vec)
    voltages = np.array(v_vec)
    currents = np.array(i_vec)
    
    # Safely convert spikes
    spikes = []
    try:
        spikes = [float(t) for t in spike_times]
    except:
        pass
    
    print(f"\nResults:")
    print(f"Voltage range: {voltages.min():.1f} to {voltages.max():.1f} mV")
    print(f"Current range: {currents.min():.6f} to {currents.max():.6f} mA/cm²")
    print(f"Spikes detected: {len(spikes)}")
    
    if voltages.max() < 100 and voltages.min() > -100:
        print("✅ Voltage remained in physiological range")
    else:
        print("❌ Voltage went outside normal range")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Voltage
    axes[0].plot(times, voltages, 'b-', linewidth=1)
    axes[0].axhline(-20, color='r', linestyle='--', alpha=0.7, label='Spike threshold')
    axes[0].axhline(-65, color='g', linestyle=':', alpha=0.7, label='Leak reversal')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_title('Membrane Potential (Stable Parameters)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mark spikes
    for spike_t in spikes:
        axes[0].axvline(spike_t, color='red', alpha=0.5, linestyle=':')
    
    # Current
    axes[1].plot(times, currents, 'g-', linewidth=1)
    axes[1].set_ylabel('Current (mA/cm²)')
    axes[1].set_title('Membrane Current from ode_neuron mechanism')
    axes[1].grid(True, alpha=0.3)
    
    # Stimulation indicator
    stim_current = np.zeros_like(times)
    stim_mask = (times >= stim.delay) & (times <= stim.delay + stim.dur)
    stim_current[stim_mask] = stim.amp
    
    axes[2].plot(times, stim_current, 'r-', linewidth=2)
    axes[2].set_ylabel('Stimulation (nA)')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_title('Applied Current Stimulation')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return len(spikes) > 0 and voltages.max() < 100

def test_minimal_network():
    """Test minimal two-neuron network with stable parameters"""
    print("\n=== Testing Minimal Two-Neuron Network ===")
    
    if not load_mechanisms():
        return
    
    # Use proven stable parameters
    initial_concentrations = [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    neuron_parameters = [
        5.0e-3, 0.01, 1.0, 0.9, 5.0e-4, 0.2, 0.1, 1.0, 0.9, 0.005, 
        0.3, 0.1, 0.0001, 0.00001, 0.0005, 0.0005, 0.0005, 0.0005, 
        0.9, 0.1, 0.1, 0.9, 0.0011, 0.0001, 0.0001, 0.00001,
        50.0, 0.1, 0.0003, -65.0, -20.0  # Conservative g_leak
    ]
    
    # Create neurons
    pre_neuron = ODENeuron("pre_stable", initial_concentrations, neuron_parameters)
    post_neuron = ODENeuron("post_stable", initial_concentrations, neuron_parameters)
    
    # Create synapse
    post_segment = post_neuron.soma(0.5)
    syn = h.ProbabilisticSyn(post_segment.parent.hoc(post_segment.x))
    
    # Create NetCon with conservative settings
    netcon = h.NetCon(pre_neuron.soma_segment.hoc._ref_v, syn, sec=pre_neuron.soma.hoc)
    netcon.weight[0] = 0.5  # Moderate weight
    netcon.delay = 2.0      # Longer delay
    netcon.threshold = -15.0  # Slightly higher threshold for stability
    
    print(f"NetCon: weight={netcon.weight[0]}, threshold={netcon.threshold}, delay={netcon.delay}")
    
    # Set pointer
    try:
        h.setpointer(post_neuron.ode_mech._ref_syn_input_activity, 'target_syn_input_activity', syn)
        print("✅ Pointer connection successful")
    except Exception as e:
        print(f"❌ Pointer failed: {e}")
        return False
    
    # Moderate stimulation
    stim = h.IClamp(pre_neuron.soma_segment.hoc)
    stim.delay = 50
    stim.dur = 100
    stim.amp = 12.0  # Strong enough to spike but not explosive
    
    # Record everything
    t_vec = h.Vector()
    pre_v_vec = h.Vector()
    post_v_vec = h.Vector()
    post_syn_vec = h.Vector()
    
    t_vec.record(h._ref_t)
    pre_v_vec.record(pre_neuron.soma_segment.hoc._ref_v)
    post_v_vec.record(post_neuron.soma_segment.hoc._ref_v)
    post_syn_vec.record(post_neuron.ode_mech._ref_syn_input_activity)
    
    # Record spikes and events
    pre_spikes = h.Vector()
    netcon_events = h.Vector()
    pre_neuron.spike_detector.record(pre_spikes)
    netcon.record(netcon_events)
    
    # Run simulation
    h.finitialize(-65)
    h.dt = 0.025
    h.tstop = 300
    
    print("Running minimal network simulation...")
    
    while h.t < h.tstop:
        h.fadvance()
        
        # Safety monitoring
        pre_v = pre_neuron.soma_segment.hoc.v
        post_v = post_neuron.soma_segment.hoc.v
        
        if abs(pre_v) > 200 or abs(post_v) > 200:
            print(f"⚠️  High voltage: pre={pre_v:.1f}, post={post_v:.1f} at t={h.t:.1f}")
            if abs(pre_v) > 500 or abs(post_v) > 500:
                print("❌ Stopping due to instability")
                break
    
    # Process results
    times = np.array(t_vec)
    pre_v = np.array(pre_v_vec)
    post_v = np.array(post_v_vec)
    post_syn = np.array(post_syn_vec)
    
    # Safely convert events
    pre_spike_times = []
    netcon_event_times = []
    
    try:
        pre_spike_times = [float(t) for t in pre_spikes]
    except:
        pass
    
    try:
        netcon_event_times = [float(t) for t in netcon_events]
    except:
        pass
    
    print(f"\nMinimal Network Results:")
    print(f"Pre-neuron V range: {pre_v.min():.1f} to {pre_v.max():.1f} mV")
    print(f"Post-neuron V range: {post_v.min():.1f} to {post_v.max():.1f} mV")
    print(f"Pre-neuron spikes: {len(pre_spike_times)}")
    print(f"NetCon events: {len(netcon_event_times)}")
    print(f"Max synaptic input: {post_syn.max():.3f}")
    
    # Check for success
    stable = (abs(pre_v.max()) < 100 and abs(post_v.max()) < 100)
    spikes = len(pre_spike_times) > 0
    transmission = len(netcon_event_times) > 0
    
    if stable and spikes and transmission:
        print("✅ SUCCESS: Stable network with synaptic transmission!")
    elif stable and spikes:
        print("⚠️  Neurons spike stably but no synaptic transmission")
    elif stable:
        print("⚠️  Network stable but no spiking")
    else:
        print("❌ Network unstable")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    axes[0].plot(times, pre_v, 'b-', linewidth=1, label='Pre-neuron')
    axes[0].axhline(netcon.threshold, color='r', linestyle='--', alpha=0.7, label=f'NetCon threshold')
    axes[0].set_ylabel('Pre V (mV)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mark spikes
    for spike_t in pre_spike_times:
        axes[0].axvline(spike_t, color='red', alpha=0.5)
    
    axes[1].plot(times, post_v, 'g-', linewidth=1, label='Post-neuron')
    axes[1].set_ylabel('Post V (mV)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(times, post_syn, 'm-', linewidth=1, label='Synaptic input')
    axes[2].set_ylabel('Synaptic Input')
    axes[2].set_xlabel('Time (ms)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Mark NetCon events
    for event_t in netcon_event_times:
        axes[2].axvline(event_t, color='orange', alpha=0.7, linestyle=':', linewidth=2)
    
    plt.suptitle('Minimal Stable Network Test')
    plt.tight_layout()
    plt.show()
    
    return stable and transmission

def main():
    """Run diagnostic tests to fix voltage explosion"""
    print("🔍 Diagnosing Voltage Explosion Issue")
    print("=" * 50)
    
    # Step 1: Find stable leak conductance
    stable_g_leak = test_leak_conductance_values()
    
    if stable_g_leak:
        print(f"\n✅ Found stable g_leak value: {stable_g_leak}")
    else:
        print("\n❌ No stable g_leak found in tested range")
        return
    
    # Step 2: Test with stable parameters
    single_stable = test_with_stable_parameters()
    
    if not single_stable:
        print("\n❌ Even conservative parameters are unstable")
        print("💡 The issue might be in your ode_neuron.mod file")
        print("   Check the BREAKPOINT block and current calculation")
        return
    
    # Step 3: Test minimal network
    network_works = test_minimal_network()
    
    if network_works:
        print("\n🎉 SUCCESS! Found working parameters:")
        print("   g_leak = 0.0003 S/cm²")
        print("   NetCon threshold = -15.0 mV")
        print("   Moderate stimulation (8-12 nA)")
        print("   Conservative synaptic weights (0.5)")
    else:
        print("\n⚠️  Single neurons work but network needs more tuning")

if __name__ == "__main__":
    main()