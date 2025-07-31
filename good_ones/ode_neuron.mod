NEURON {
    SUFFIX ode_neuron
    RANGE g_leak, e_leak
    RANGE P, B, p75, TrkB, p75_pro, p75_B, TrkB_B, TrkB_pro, tPA
    RANGE ksP, k_cleave, k_cleave_variable, k_p75_pro_on, k_p75_pro_off, k_degP, k_TrkB_pro_on, k_TrkB_pro_off
    RANGE k_TrkB_B_on, k_TrkB_B_off, k_degB, k_p75_B_on, k_p75_B_off, k_degR1, k_degR2
    RANGE k_int_p75_pro, k_int_p75_B, k_int_TrkB_B, k_int_TrkB_pro, aff_p75_pro
    RANGE aff_p75_B, aff_TrkB_pro, aff_TrkB_B, k_deg_tPA, ks_tPA, ks_p75, ks_TrkB
    RANGE activity_level
    RANGE v_threshold_spike
    RANGE growth_strength, apop_strength
    RANGE syn_input_activity
    RANGE tau_activity, activity_gain
    THREADSAFE
}

UNITS {
    (molar) = (1)
    (mM) = (millimolar)
    (nM) = (nanomolar)
    (uM) = (micromolar)
    (pM) = (picomolar)
    (mS) = (millisiemens)
    (MS) = (/micromolar/second)
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uF) = (microfarad)
    (S) = (siemens)
}

PARAMETER {
    : Parameters associated with membrane properties
    g_leak = 0.0001 (S/cm2) : Leak conductance
    e_leak = -65 (mV)       : Leak reversal potential
    v_threshold_spike = -20 (mV) : Threshold for spike detection (constant)

    : Your existing ODE parameters (ensure these match your Python param list order)
    ksP = 5.0e-3 (uM/s)   : synthesis rate of proBDNF
    k_cleave = 0.01 (/s)    : rate of proBDNF cleavage into BDNF
    k_p75_pro_on = 1.0 (MS)    : proBDNF binding to p75
    k_p75_pro_off = 0.9 (/s)    : proBDNF unbinding from p75
    k_degP = 5.0e-4 (/s)   : proBDNF degradation
    k_TrkB_pro_on = 0.2 (MS)    : proBDNF binding to TrkB
    k_TrkB_pro_off = 0.1 (/s)   : proBDNF unbinding from TrkB
    k_TrkB_B_on = 1.0 (MS)    : BDNF binding to TrkB
    k_TrkB_B_off = 0.9 (/s)    : BDNF unbinding from TrkB
    k_degB = 0.005 (/s)    : BDNF degradation
    k_p75_B_on = 0.3 (MS)    : BDNF binding to p75
    k_p75_B_off = 0.1 (/s)   : BDNF unbinding from p75
    k_degR1 = 0.0001 (/s)   : p75 degradation
    k_degR2 = 0.00001 (/s)   : TrkB degradation
    k_int_p75_pro = 0.0005 (/s)    : proBDNF-p75 internalization
    k_int_p75_B = 0.0005 (/s)    : BDNF-p75 internalization
    k_int_TrkB_B = 0.0005 (/s)    : BDNF-TrkB internalization
    k_int_TrkB_pro = 0.0005 (/s)    : proBDNF-TrkB internalization
    aff_p75_pro = 0.9    : affinity of proBDNF for p75
    aff_p75_B = 0.1    : affinity of BDNF for p75
    aff_TrkB_pro = 0.1    : affinity of proBDNF for TrkB
    aff_TrkB_B = 0.9    : affinity of BDNF for TrkB
    k_deg_tPA = 0.0011 (/s)   : degradation rate of tPA - slow degradation
    ks_tPA = 0.0001 (uM/s)    : synthesis rate of tPA
    ks_p75 = 0.0001 (uM/s)    : synthesis rate of p75 - small value to maintain baseline
    ks_TrkB = 0.00001 (uM/s)    : synthesis rate of TrkB - small value to maintain baseline
    tau_activity = 50 (ms) : Time constant for activity_level decay
    activity_gain = 0.1 : How much one synaptic event boosts activity_level
}

ASSIGNED {
    v (mV) : Membrane potential
    
    ks_P_variable (uM/s)
    ks_tPA_variable (uM/s)
    growth_strength 
    apop_strength 
    syn_input_activity : Input from synapses
    k_cleave_variable (1/s) : Cleavage rate of proBDNF into BDNF
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
    activity_level : Activity level factor
}

INITIAL {
    P = 0.2    : Initial concentration of proBDNF
    B = 0.0    : Initial concentration of BDNF
    p75 = 1.0    : Initial concentration of p75 receptor
    TrkB = 1.0    : Initial concentration of TrkB receptor
    p75_pro = 0.0    : Initial concentration of proBDNF bound to p75
    p75_B = 0.0    : Initial concentration of BDNF bound to p75
    TrkB_B = 0.0    : Initial concentration of BDNF bound to TrkB
    TrkB_pro = 0.0    : Initial concentration of proBDNF bound to TrkB
    tPA = 0.1    : Initial concentration of tPA
    activity_level = 0.0 : Initial activity level
    syn_input_activity = 0.0 : Initialize syn_input_activity
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    
    : Calculate current contributed by this mechanism (leak current)
    k_cleave_variable = k_cleave * (1 + activity_level * 4)
    ks_P_variable = ksP * (1 + activity_level)
    ks_tPA_variable = ks_tPA * (1 + activity_level*4)

    growth_strength = (Hill(TrkB_B, 0.05, 2) + Hill(TrkB_pro, 0.02, 2))/2
    apop_strength = (Hill(p75_pro, 0.02, 2) + Hill(p75_B, 0.02, 2))/2

}

DERIVATIVE states {
    P' = ks_P_variable - k_cleave_variable * tPA * P - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degP * P
        
    B' = k_cleave_variable * tPA * P - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degB * B
    
    p75' = ks_p75 * (1 + activity_level * 0.001) - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degR1 * p75
        
    TrkB' = ks_TrkB * (1 + activity_level * 0.002) - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degR2 * TrkB
        
    p75_pro' = k_p75_pro_on * aff_p75_pro * P * p75 - k_p75_pro_off * p75_pro - k_int_p75_pro * p75_pro
      
    p75_B' = k_p75_B_on * aff_p75_B * B * p75 - k_p75_B_off * p75_B - k_int_p75_B * p75_B
        
    TrkB_B' = k_TrkB_B_on * aff_TrkB_B * B * TrkB - k_TrkB_B_off * TrkB_B - k_int_TrkB_B * TrkB_B
        
    TrkB_pro' = k_TrkB_pro_on * aff_TrkB_pro * P * TrkB - k_TrkB_pro_off * TrkB_pro - k_int_TrkB_pro * TrkB_pro
        
    tPA' = ks_tPA_variable - k_deg_tPA * tPA


    activity_level' = -activity_level / tau_activity + syn_input_activity
}

FUNCTION Hill(C, KD, n) {
    Hill = C^n / (KD^n + C^n)
}