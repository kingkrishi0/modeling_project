: mods/ode_neuron.mod
NEURON {
    SUFFIX ode_neuron
    RANGE g_leak, e_leak, cm
    RANGE P, B, p75, TrkB, p75_pro, p75_B, TrkB_B, TrkB_pro, tPA
    RANGE ksP, k_cleave, k_p75_pro_on, k_p75_pro_off, k_degP, k_TrkB_pro_on, k_TrkB_pro_off
    RANGE k_TrkB_B_on, k_TrkB_B_off, k_degB, k_p75_B_on, k_p75_B_off, k_degR1, k_degR2
    RANGE k_int_p75_pro, k_int_p75_B, k_int_TrkB_B, k_int_TrkB_pro, aff_p75_pro
    RANGE aff_p75_B, aff_TrkB_pro, aff_TrkB_B, k_deg_tPA, ks_tPA, ks_p75, ks_TrkB
    RANGE activity_level
    RANGE v_threshold_spike
    RANGE growth_strength, apop_strength
    RANGE syn_input_activity
    THREADSAFE
}

UNITS {
    (molar) = (1)
    (mM) = (millimolar)
    (nM) = (nanomolar)
    (uM) = (micromolar)
    (pM) = (picomolar)
    (mS) = (millisiemens)
    (MS) = (1/ micromolar 1/ second)
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uF) = (microfarad)
}

PARAMETER {
    : Parameters associated with membrane properties
    cm = 1 (uF/cm2)         : Membrane capacitance
    g_leak = 0.0001 (S/cm2) : Leak conductance
    e_leak = -65 (mV)       : Leak reversal potential
    v_threshold_spike = -20 (mV) : Threshold for spike detection (constant)

    : Your existing ODE parameters (ensure these match your Python param list order)
    ksP = 5.0e-3 (uM/s)   : synthesis rate of proBDNF
    k_cleave = 0.01 (1/s)    : rate of proBDNF cleavage into BDNF
    k_p75_pro_on = 1.0 (MS)    : proBDNF binding to p75
    k_p75_pro_off = 0.9 (1/s)    : proBDNF unbinding from p75
    k_degP = 5.0e-4 (1/s)   : proBDNF degradation
    k_TrkB_pro_on = 0.2 (MS)    : proBDNF binding to TrkB
    k_TrkB_pro_off = 0.1 (1/s)   : proBDNF unbinding from TrkB
    k_TrkB_B_on = 1.0 (MS)    : BDNF binding to TrkB
    k_TrkB_B_off = 0.9 (1/s)    : BDNF unbinding from TrkB
    k_degB = 0.005 (1/s)    : BDNF degradation
    k_p75_B_on = 0.3 (MS)    : BDNF binding to p75
    k_p75_B_off = 0.1 (1/s)   : BDNF unbinding from p75
    k_degR1 = 0.0001 (1/s)   : p75 degradation
    k_degR2 = 0.00001 (1/s)   : TrkB degradation
    k_int_p75_pro = 0.0005 (1/s)    : proBDNF-p75 internalization
    k_int_p75_B = 0.0005 (1/s)    : BDNF-p75 internalization
    k_int_TrkB_B = 0.0005 (1/s)    : BDNF-TrkB internalization
    k_int_TrkB_pro = 0.0005 (1/s)    : proBDNF-TrkB internalization
    aff_p75_pro = 0.9    : affinity of proBDNF for p75
    aff_p75_B = 0.1    : affinity of BDNF for p75
    aff_TrkB_pro = 0.1    : affinity of proBDNF for TrkB
    aff_TrkB_B = 0.9    : affinity of BDNF for TrkB
    k_deg_tPA = 0.0011 (1/s)   : degradation rate of tPA - slow degradation
    ks_tPA = 0.0001 (uM/s)    : synthesis rate of tPA
    ks_p75 = 0.0001 (uM/s)    : synthesis rate of p75 - small value to maintain baseline
    ks_TrkB = 0.00001 (uM/s)    : synthesis rate of TrkB - small value to maintain baseline
    
    : Parameters for activity level dynamics
    tau_activity = 50 (ms) : Time constant for activity_level decay
    activity_gain = 0.1 (unitless) : How much one synaptic event boosts activity_level
}

ASSIGNED {
    v (mV) : Membrane potential (assigned by NEURON's solver, not a state of this mechanism)
    ica (mA/cm2) : if you use Ca ions.
    i (mA/cm2) : Current contributed by this mechanism (e.g., leak current)
    
    ks_P_variable (uM/s)
    ks_tPA_variable (uM/s)
    growth_strength
    apop_strength
    syn_input_activity (unitless) : 
    : REMOVED: syn_input_activity is a POINTER from ProbabilisticSyn, not ASSIGNED here
}

STATE {
    P (uM)    : proBDNF concentration
    B (uM)    : BDNF concentration
    p75 (uM)    : p75 receptor concentration
    TrkB (uM)    : TrkB receptor concentration
    p75_pro (uM)    : proBDNF bound to p75
    p75_B (uM)    : BDNF bound to p75
    TrkB_B (uM)    : BDNF bound to TrkB
    TrkB_pro (uM)    : proBDNF bound to TrkB
    tPA (uM)    : tPA concentration
    
    activity_level (unitless) : Activity level factor (now a state variable)
}

INITIAL {
    P = 0.2 (uM)    : Initial concentration of proBDNF
    B = 0.0 (uM)    : Initial concentration of BDNF
    p75 = 1.0 (uM)    : Initial concentration of p75 receptor
    TrkB = 1.0 (uM)    : Initial concentration of TrkB receptor
    p75_pro = 0.0 (uM)    : Initial concentration of proBDNF bound to p75
    p75_B = 0.0 (uM)    : Initial concentration of BDNF bound to p75
    TrkB_B = 0.0 (uM)    : Initial concentration of BDNF bound to TrkB
    TrkB_pro = 0.0 (uM)    : Initial concentration of proBDNF bound to TrkB
    tPA = 0.1 (uM)    : Initial concentration of tPA
    
    activity_level = 0.0 : Initial activity level
    syn_input_activity = 0.0 : Initialize syn_input_activity (it's a RANGE, so can be initialized here)

    : Initialize assigned values
    ica = 0
    i = 0
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    
    : Calculate current contributed by *this* mechanism (leak current)
    i = g_leak * (e_leak - v) : Standard leak current for a density mechanism
    : NEURON's internal solver will sum this 'i' with NONSPECIFIC_CURRENT i_inj and other currents.

    ks_P_variable = ksP * (1 + activity_level)  : Adjusted synthesis rate of proBDNF based on activity level
    ks_tPA_variable = ks_tPA * (1 + activity_level)  : Adjusted synthesis rate of tPA based on activity level

    growth_strength = (Hill(TrkB_B, 0.05 (uM), 2) + Hill(TrkB_pro, 0.02 (uM), 2))/2 : Growth strength based on TrkB-ligand concentrations
    apop_strength = (Hill(p75_pro, 0.02 (uM), 2) + Hill(p75_B, 0.02 (uM), 2))/2 : Apoptosis strength based on p75-ligand concentrations
}

DERIVATIVE states {
    : ODEs for biochemical concentrations and activity_level
    P' = ks_P_variable - k_cleave * tPA * P - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degP * P
        
    B' = k_cleave * tPA * P - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degB * B
    
    p75' = ks_p75 - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degR1 * p75
        
    TrkB' = ks_TrkB - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degR2 * TrkB
        
    p75_pro' = k_p75_pro_on * aff_p75_pro * P * p75 - k_p75_pro_off * p75_pro - k_int_p75_pro * p75_pro
      
    p75_B' = k_p75_B_on * aff_p75_B * B * p75 - k_p75_B_off * p75_B - k_int_p75_B * p75_B
        
    TrkB_B' = k_TrkB_B_on * aff_TrkB_B * B * TrkB - k_TrkB_B_off * TrkB_B - k_int_TrkB_B * TrkB_B
        
    TrkB_pro' = k_TrkB_pro_on * aff_TrkB_pro * P * TrkB - k_TrkB_pro_off * TrkB_pro - k_int_TrkB_pro * TrkB_pro
        
    tPA' = ks_tPA_variable - k_deg_tPA * tPA

    activity_level' = -activity_level / tau_activity + syn_input_activity : `syn_input_activity` is a 'kick' rate
    
}

FUNCTION Hill(C (uM), KD (uM), n) {
    Hill = C^n / (KD^n + C^n)
}