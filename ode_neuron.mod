UNITS {
    (molar) = (1)
    (mM) = (millimolar)
    (nM) = (nanomolar)
    (uM) = (micromolar)
    (pM) = (picomolar)
    (mS) = (millisiemens)
    (MS) = (1/ micromolar 1/ second)
}

PARAMETER {
    ksP = 5.0e-3 (uM/s)   : ksP (synthesis rate of proBDNF)
    k_cleave = 0.01 (1/s)    : k_cleave (rate of proBDNF cleavage into BDNF)
    k_p75_pro_on = 1.0 (MS)    : k_p75_pro_on (proBDNF binding to p75)
    k_p75_pro_off = 0.9 (1/s)    : k_p75_pro_off (proBDNF unbinding from p75)
    k_degP = 5.0e-4 (1/s)   : k_degP (proBDNF degradation )
    k_TrkB_pro_on = 0.2 (MS)    : k_TrkB_pro_on (proBDNF binding to TrkB)
    k_TrkB_pro_off = 0.1 (1/s)   : k_TrkB_pro_off (proBDNF unbinding from TrkB)
    k_TrkB_B_on = 1.0 (MS)    : k_TrkB_B_on (BDNF binding to TrkB)
    k_TrkB_B_off = 0.9 (1/s)    : k_TrkB_B_off (BDNF unbinding from TrkB)
    k_degB = 0.005 (1/s)    : k_degB (BDNF degradation)
    k_p75_B_on = 0.3 (MS)    : k_p75_B_on (BDNF binding to p75)
    k_p75_B_off = 0.1 (1/s)   : k_p75_B_off (BDNF unbinding from p75)
    k_degR1 = 0.0001 (1/s)   : k_degR1 (p75 degradation)
    k_degR2 = 0.00001 (1/s)   : k_degR2 (TrkB degradation)
    k_int_p75_pro = 0.0005 (1/s)    : k_int_p75_pro (proBDNF-p75 internalization)
    k_int_p75_B = 0.0005 (1/s)    : k_int_p75_B (BDNF-p75 internalization)
    k_int_TrkB_B = 0.0005 (1/s)    : k_int_TrkB_B (BDNF-TrkB internalization)
    k_int_TrkB_pro = 0.0005 (1/s)    : k_int_TrkB_pro (proBDNF-TrkB internalization)
    aff_p75_pro = 0.9    : aff_p75_pro (affinity of proBDNF for p75)
    aff_p75_B = 0.1    : aff_p75_B (affinity of BDNF for p75)
    aff_TrkB_pro = 0.1    : aff_TrkB_pro (affinity of proBDNF for TrkB)
    aff_TrkB_B = 0.9    : aff_TrkB_B (affinity of BDNF for TrkB)
    k_deg_tPA = 0.0011 (1/s)   :k_deg_tPA (degradation rate of tPA) - slow degradation
    ks_tPA = 0.0001 (uM/s)    : ks_tPA (synthesis rate of tPA)
    : NEW PARAMETERS FOR BIOLOGICAL ACCURACY
    ks_p75 = 0.0001 (uM/s)    : ks_p75 (synthesis rate of p75) - small value to maintain baseline
    ks_TrkB = 0.00001 (uM/s)    : ks_TrkB (synthesis rate of TrkB) - small value to maintain baseline
    activity_level = 1.0   : activity level factor (default is 1, can be adjusted)
}

NEURON {
    SUFFIX ode_neuron
    USEION ca READ ica WRITE ica
    NONSPECIFIC_CURRENT i
    RANGE P, B, p75, TrkB, p75_pro, p75_B, TrkB_B, TrkB_pro, tPA
    RANGE ksP, k_cleave, k_p75_pro_on, k_p75_pro_off, k_degP, k_TrkB_pro_on, k_TrkB_pro_off
    RANGE k_TrkB_B_on, k_TrkB_B_off, k_degB, k_p75_B_on, k_p75_B_off, k_degR1, k_degR2
    RANGE k_int_p75_pro, k_int_p75_B, k_int_TrkB_B, k_int_TrkB_pro, aff_p75_pro
    RANGE aff_p75_B, aff_TrkB_pro, aff_TrkB_B, k_deg_tPA, ks_tPA, ks_p75, ks_TrkB

    RANGE activity_level

    RANGE growth_strength, apop_strength
}

ASSIGNED {
    ks_P_variable (uM/s)  : Adjusted synthesis rate of proBDNF based on activity level
    ks_tPA_variable (uM/s)  : Adjusted synthesis rate of tPA based on activity level
    growth_strength  : Growth strength factor
    apop_strength    : Apoptosis strength factor
    ica (mA/cm2)
    i (mA/cm2)
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
    ica = 0
    i = 0
}

BREAKPOINT {
    SOLVE integrate METHOD derivimplicit
    ks_P_variable = ksP * activity_level  : Adjusted synthesis rate of proBDNF based on activity level
    ks_tPA_variable = ks_tPA * activity_level  : Adjusted synthesis rate of tPA based on activity level

    growth_strength = (Hill(TrkB_B, 0.05 (uM), 2) + Hill(TrkB_pro, 0.02 (uM), 2))/2 : Growth strength based on proBDNF and BDNF concentrations
    apop_strength = (Hill(p75_pro, 0.02 (uM), 2) + Hill(p75_B, 0.02 (uM), 2))/2 : Apoptosis strength based on proBDNF concentration

   
}

DERIVATIVE integrate {
    P' = ks_P_variable - k_cleave * tPA * P - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degP * P
        
    B' = k_cleave * tPA * P - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degB * B
    
    p75' = ks_p75 - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degR1 * p75
        
    TrkB' = ks_TrkB - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degR2 * TrkB
        
    p75_pro' = k_p75_pro_on * aff_p75_pro * P * p75 - k_p75_pro_off * p75_pro - k_int_p75_pro * p75_pro
        
    p75_B' = k_p75_B_on * aff_p75_B * B * p75 - k_p75_B_off * p75_B - k_int_p75_B * p75_B
        
    TrkB_B' = k_TrkB_B_on * aff_TrkB_B * B * TrkB - k_TrkB_B_off * TrkB_B - k_int_TrkB_B * TrkB_B
        
    TrkB_pro' = k_TrkB_pro_on * aff_TrkB_pro * P * TrkB - k_TrkB_pro_off * TrkB_pro - k_int_TrkB_pro * TrkB_pro
        
    tPA' = ks_tPA_variable - k_deg_tPA * tPA 
}



FUNCTION Hill(C (uM), KD (uM), n) {
    : C: Concentration (uM)
    : KD: Half-maximal concentration (uM)
    : n: Hill coefficient (dimensionless)
    
    Hill = C^n / (KD^n + C^n)
}

