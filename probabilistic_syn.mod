// mods/probabilistic_syn.mod
// Based on NEURON's ExpSyn, but with probabilistic transmission
// and an event sent to the postsynaptic ODENeuron for activity level update.

NEURON {
    POINT_PROCESS ProbabilisticSyn
    RANGE tau, e, i, prob_factor
    NONSPECIFIC_CURRENT i
    GLOBAL _rng_seed : For reproducibility of random numbers
    BBCOREPOINTER netcon_event_target : Pointer to a NetCon for sending event to ODENeuron
}

PARAMETER {
    tau = 2 (ms)     : decay time constant for current
    e = 0 (mV)       : reversal potential
    prob_factor = 0.5 : Base probability scaling (0 to 1)
    _rng_seed = 1234  : Seed for random number generator
}

ASSIGNED {
    v (mV)
    i (nA)
    g (nS)           : conductance
    event_weight (unitless) : incoming NetCon weight (from presynaptic neuron)
    random_num (unitless) : random number for probabilistic check
    rng           : random number generator object
}

STATE {
    G (nS)           : state variable for conductance
}

INITIAL {
    G = 0
    i = 0
    set_seed(_rng_seed) // Initialize RNG for each synapse instance
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = g * (v - e)
    g = G
}

DERIVATIVE state {
    G' = -G / tau
}

NET_RECEIVE(w (unitless)) {
    : w is the NetCon weight from the presynaptic spike detector.
    : This 'w' is the synaptic weight modulated by your learning rules.
    event_weight = w
    
    : Get a random number between 0 and 1
    random_num = uniform(0, 1)

    : Calculate probability of transmission based on synaptic weight.
    : A simple linear scaling: prob = weight * prob_factor
    : Clamped between 0 and 1.
    prob_threshold = event_weight * prob_factor * 2 // Multiply by 2 as initial_syn_weight is 0.5
    prob_threshold = min(1.0, max(0.0, prob_threshold)) // Clamp between 0 and 1

    if (random_num < prob_threshold) {
        // If transmission is successful:
        // 1. Generate a Postsynaptic Current (PSC)
        G = G + event_weight * 0.05 // Scale weight to a reasonable conductance change (adjust 0.05)
                                    // This 0.05 (nS/unit) determines PSC amplitude. Tune this.
        
        // 2. Send an event to the postsynaptic ODENeuron for activity level update
        // The '1.0' argument is the 'weight_from_syn_event' in ODENeuron's NET_RECEIVE.
        // This signifies a "successful event" to the ODE.
        net_event(netcon_event_target, t + 0.1, 1.0) // 0.1ms fixed synaptic delay
    }
}

PROCEDURE set_seed(seed) {
    rng = new Random()
    rng.MCellRan4(seed, seed * 2) // MCellRan4 for better statistical properties
}

FUNCTION uniform(min, max) {
    return rng.uniform(min, max)
}