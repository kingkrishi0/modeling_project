import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

""""
# Amounts
P(t) = proBDNF
B(t) = BDNF
p75(t) = free p75
TrkB(t) = free TrkB
p75_pro(t) = proBDNF-p75 complex
p75_B(t) = BDNF-p75 complex
TrkB_B(t) = BDNF-TrkB complex
TrkB_pro(t) = proBDNF-TrkB complex
tPA(t) = tPA enzymes

# K's
ksP = synthesis rate of proBDNF M/time
k_cleave = rate of proBDNF cleavage into BDNF (frequency) 1/time
k_p75_pro_on = rate of proBDNF to bind to p75 1/(M*time)
k_p75_pro_off = rate of proBDNF to unbind from the proBDNF-p75 1/time
k_degP = rate of proBDNF degradation 1/time
k_TrkB_B_on = rate of BDNF to bind to TrkB 1/(M*time)
k_TrkB_B_off = rate of BDNF to unbind from the BDNF-TrkB complex 1/time
k_degB = rate of BDNF degradation 1/time
k_degR1 = rate of p75 degradation 1/time
k_degR2 = rate of TrkB degradation 1/time
k_int_p75_pro = rate of proBDNF-p75 complex internalization 1/time
k_int_p75_B = rate of BDNF-p75 complex internalization 1/time
k_int_TrkB_B = rate of BDNF-TrkB complex internalization 1/time
k_int_TrkB_pro = rate of proBDNF-TrkB complex internalization 1/time

# Affinities
aff_p75_pro = 1.0  # Affinity of proBDNF for p75, assumed constant for simplicity
aff_p75_B = 1.0  # Affinity of BDNF for p75, assumed constant for simplicity
aff_TrkB_pro = 1.0  # Affinity of proBDNF for TrkB, assumed constant for simplicity
aff_TrkB_B = 1.0  # Affinity of BDNF for TrkB, assumed constant for simplicity

# Parameter Order
# ksP
# k_cleave
# k_p75_pro_on
# k_p75_pro_off
# k_degP
# k_TrkB_pro_on
# k_TrkB_pro_off
# k_TrkB_B_on
# k_TrkB_B_off
# k_degB
# k_p75_B_on
# k_p75_B_off
# k_degR1
# k_degR2
# k_int_p75_pro
# k_int_p75_B
# k_int_TrkB_B
# k_int_TrkB_pro
# aff_p75_pro
# aff_p75_B
# aff_TrkB_pro
# aff_TrkB_B
# k_deg_tPA
# ks_tPA
# ks_p75
# ks_TrkB
"""

# Define the ODE system
def dYdt(t, y, params):
    P, B, p75, TrkB, p75_pro, p75_B, TrkB_B, TrkB_pro, tPA = y
    ksP, k_cleave, k_p75_pro_on, k_p75_pro_off, k_degP, k_TrkB_pro_on, k_TrkB_pro_off, \
    k_TrkB_B_on, k_TrkB_B_off, k_degB, k_p75_B_on, k_p75_B_off, k_degR1, k_degR2, \
    k_int_p75_pro, k_int_p75_B, k_int_TrkB_B, k_int_TrkB_pro, aff_p75_pro, \
    aff_p75_B, aff_TrkB_pro, aff_TrkB_B, k_deg_tPA, ks_tPA, ks_p75, ks_TrkB= params

    activity_level = 1.0 + 0.5 * np.sin(2 * np.pi * t / 20)
    #activity_level = np.random.uniform(0.5, 1.5)

    ksP_variable = ksP * activity_level

    dP = ksP_variable - k_cleave * tPA * P - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degP * P
    
    dB = k_cleave * tPA * P - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degB * B
    
    dp75 = ks_p75 - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degR1 * p75
    
    dTrkB = ks_TrkB - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degR2 * TrkB
    
    dp75_pro = k_p75_pro_on * aff_p75_pro * P * p75 - k_p75_pro_off * p75_pro - k_int_p75_pro * p75_pro
    
    dp75_B = k_p75_B_on * aff_p75_B * B * p75 - k_p75_B_off * p75_B - k_int_p75_B * p75_B
    
    dTrkB_B = k_TrkB_B_on * aff_TrkB_B * B * TrkB - k_TrkB_B_off * TrkB_B - k_int_TrkB_B * TrkB_B
    
    dTrkB_pro = k_TrkB_pro_on * aff_TrkB_pro * P * TrkB - k_TrkB_pro_off * TrkB_pro - k_int_TrkB_pro * TrkB_pro
    
    dtPA = ks_tPA * activity_level + 0.1 * TrkB_B - k_deg_tPA * tPA#-k_cleave * tPA * P + k_TrkB_B_on * aff_TrkB_B * B * TrkB

    return [dP, dB, dp75, dTrkB, dp75_pro, dp75_B, dTrkB_B, dTrkB_pro, dtPA]

# Initial concentrations
y0 = [
    0.2,   # P: proBDNF (low initial, newly synthesized)
    0.0,   # B: BDNF (none at t=0, must be cleaved from proBDNF)
    1.0,   # p75: free p75 receptor (abundant)
    1.0,   # TrkB: free TrkB receptor (abundant)
    0.0,   # p75_pro: proBDNF-p75 complex (none at t=0)
    0.0,   # p75_B: BDNF-p75 complex (none at t=0)
    0.0,   # TrkB_B: BDNF-TrkB complex (none at t=0)
    0.0,   # TrkB_pro: proBDNF-TrkB complex (none at t=0)
    1.0     # tPA: tPA enzyme (moderate, present to allow cleavage)
]

# Parameters
params = [
    0.5,   # ksP (synthesis rate of proBDNF)
    0.3,    # k_cleave (rate of proBDNF cleavage into BDNF)
    1.0,    # k_p75_pro_on (proBDNF binding to p75)
    0.1,    # k_p75_pro_off (proBDNF unbinding from p75)
    0.01,   # k_degP (proBDNF degradation)
    0.5,    # k_TrkB_pro_on (proBDNF binding to TrkB)
    0.05,   # k_TrkB_pro_off (proBDNF unbinding from TrkB)
    1.0,    # k_TrkB_B_on (BDNF binding to TrkB)
    0.1,    # k_TrkB_B_off (BDNF unbinding from TrkB)
    0.36,    # k_degB (BDNF degradation)
    0.8,    # k_p75_B_on (BDNF binding to p75)
    0.08,   # k_p75_B_off (BDNF unbinding from p75)
    0.01,   # k_degR1 (p75 degradation)
    0.01,   # k_degR2 (TrkB degradation)
    0.1,    # k_int_p75_pro (proBDNF-p75 internalization)
    0.1,    # k_int_p75_B (BDNF-p75 internalization)
    0.1,    # k_int_TrkB_B (BDNF-TrkB internalization)
    0.1,    # k_int_TrkB_pro (proBDNF-TrkB internalization)
    0.9,    # aff_p75_pro (affinity of proBDNF for p75)
    0.1,    # aff_p75_B (affinity of BDNF for p75)
    0.1,    # aff_TrkB_pro (affinity of proBDNF for TrkB)
    0.9,    # aff_TrkB_B (affinity of BDNF for TrkB)
    0.4,    #k_deg_tPA (degradation rate of tPA) - slow degradation
    0.3,    # ks_tPA (synthesis rate of tPA)
    # NEW PARAMETERS FOR BIOLOGICAL ACCURACY
    0.1,    # ks_p75 (synthesis rate of p75) - small value to maintain baseline
    0.05,    # ks_TrkB (synthesis rate of TrkB) - small value to maintain baseline
]

# Time span
t_span = (0, 50)
t_eval = np.linspace(*t_span, 1000)

# Solve
solution = solve_ivp(dYdt, t_span, y0, args=(params,), t_eval=t_eval)

# Plot
# P, B, p75, TrkB, p75_pro, p75_B, TrkB_B, TrkB_pro, tPA
plt.figure(figsize=(10,6))
for i, label in enumerate(['proBDNF', 'BDNF', 'p75', 'TrkB', 'p75-proBDNF', 'p75-BDNF', 'TrkB_BDNF', 'TrkB_proBDNF', 'tPA']):
    plt.plot(solution.t, solution.y[i], label=label)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Dynamics of proBDNF/BDNF and Receptors')
plt.grid()
plt.show()


plt.figure(figsize=(10,6))
for i, label in enumerate(['proBDNF', 'BDNF']):
    plt.plot(solution.t, solution.y[i], label=label)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Dynamics of proBDNF/BDNF and Receptors')
plt.grid()
plt.show()
