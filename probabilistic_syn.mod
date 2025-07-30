: Based on NEURON's ExpSyn, but with probabilistic transmission
: and directly writes to postsynaptic ODENeuron's activity input variable.

NEURON {
    POINT_PROCESS ProbabilisticSyn
    RANGE tau, e, i, prob_factor, activity_pulse_strength
    NONSPECIFIC_CURRENT i
    POINTER target_syn_input_activity
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (nS) = (nanosiemens)
}

PARAMETER {
    tau = 2 (ms)     : decay time constant for current
    e = 0 (mV)       : reversal potential
    prob_factor = 0.5 : Base probability scaling (0 to 1)
    activity_pulse_strength = 1.0 : How much a single successful event boosts syn_input_activity
}

ASSIGNED {
    v (mV)
    i (nA)
    g (nS)           : conductance
    target_syn_input_activity : POINTER to postsynaptic activity variable
}

STATE {
    G (nS)           : state variable for conductance
}

INITIAL {
    G = 0
    i = 0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    g = G
    i = g * (v - e)
}

DERIVATIVE state {
    G' = -G / tau
}

NET_RECEIVE(w) {
    LOCAL event_weight, random_num, prob_threshold
    
    : w is the NetCon weight from the presynaptic spike detector
    event_weight = w
    
    : Get a random number between 0 and 1
    random_num = scop_random()

    : Calculate probability of transmission based on synaptic weight
    prob_threshold = event_weight * prob_factor * 2 
    if (prob_threshold > 1.0) {
        prob_threshold = 1.0
    }
    if (prob_threshold < 0.0) {
        prob_threshold = 0.0
    }

    if (random_num < prob_threshold) {
        : If transmission is successful:
        : 1. Generate a Postsynaptic Current (PSC)
        G = G + event_weight * 0.05
        
        : 2. Write to the postsynaptic ODENeuron's syn_input_activity variable
        target_syn_input_activity = target_syn_input_activity + activity_pulse_strength
    }
}