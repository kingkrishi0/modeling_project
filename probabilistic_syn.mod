// mods/probabilistic_syn.mod
// Based on NEURON's ExpSyn, but with probabilistic transmission
// and directly writes to postsynaptic ODENeuron's activity input variable.

NEURON {
    POINT_PROCESS ProbabilisticSyn
    RANGE tau, e, i, prob_factor
    NONSPECIFIC_CURRENT i
    GLOBAL _rng_seed : For reproducibility of random numbers
    POINTER target_syn_input_activity : Pointer to the postsynaptic ODENeuron's syn_input_activity
}

PARAMETER {
    tau = 2 (ms)     : decay time constant for current
    e = 0 (mV)       : reversal potential
    prob_factor = 0.5 : Base probability scaling (0 to 1)
    _rng_seed = 1234  : Seed for random number generator
    activity_pulse_strength = 1.0 : How much a single successful event boosts syn_input_activity
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
    prob_threshold = event_weight * prob_factor * 2 
    prob_threshold = min(1.0, max(0.0, prob_threshold)) 

    if (random_num < prob_threshold) {
        // If transmission is successful:
        // 1. Generate a Postsynaptic Current (PSC)
        G = G + event_weight * 0.05 // Scale weight to a reasonable conductance change (adjust 0.05)
                                    // This 0.05 (nS/unit) determines PSC amplitude. Tune this.
        
        // 2. Directly write to the postsynaptic ODENeuron's syn_input_activity variable
        if (target_syn_input_activity) { // Check if pointer is valid
            target_syn_input_activity = target_syn_input_activity + activity_pulse_strength
        }
    }
}

PROCEDURE set_seed(seed) {
    rng = new Random()
    rng.MCellRan4(seed, seed * 2) // MCellRan4 for better statistical properties
}

FUNCTION uniform(min, max) {
    return rng.uniform(min, max)
}