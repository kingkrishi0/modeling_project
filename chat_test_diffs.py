import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

""""
# Amounts
P(t) = proBDNF molar
B(t) = BDNF molar
p75(t) = free p75 molar
TrkB(t) = free TrkB molar
p75_pro(t) = proBDNF-p75 complex molar
p75_B(t) = BDNF-p75 complex molar
TrkB_B(t) = BDNF-TrkB complex molar
TrkB_pro(t) = proBDNF-TrkB complex molar
tPA(t) = tPA enzymes molar

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
class ODEModel:
    def __init__(self, params, y0, t_span):
        self.params = params
        self.y0 = y0
        self.t_span = t_span
        self.solution = None

    def hill_function(self, C, KD, n):
        return C**n / (KD**n + C**n)
    
    def dYdt(self, t, y):
        P, B, p75, TrkB, p75_pro, p75_B, TrkB_B, TrkB_pro, tPA = y
        ksP, k_cleave, k_p75_pro_on, k_p75_pro_off, k_degP, k_TrkB_pro_on, k_TrkB_pro_off, \
        k_TrkB_B_on, k_TrkB_B_off, k_degB, k_p75_B_on, k_p75_B_off, k_degR1, k_degR2, \
        k_int_p75_pro, k_int_p75_B, k_int_TrkB_B, k_int_TrkB_pro, aff_p75_pro, \
        aff_p75_B, aff_TrkB_pro, aff_TrkB_B, k_deg_tPA, ks_tPA, ks_p75, ks_TrkB= self.params


        burst_duration = 50  # seconds of high activity
        inter_burst_interval = 100 # seconds between the start of bursts (period of the burst pattern)
        high_activity_value = 2.0 # The 'level' during a burst
        low_activity_value = 0.01 # The 'level' during 'no activity' (a small baseline to avoid zero multiplication)

        # Determine if 't' falls within a burst period
        # The modulo operator (%) helps create a repeating pattern
        time_within_cycle = t % inter_burst_interval

        if time_within_cycle < burst_duration:
            activity_level = high_activity_value
        else:
            activity_level = low_activity_value
        ksP_variable = ksP * activity_level

        dP = ksP_variable - k_cleave * tPA * P - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degP * P
        
        dB = k_cleave * tPA * P - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degB * B
        
        dp75 = ks_p75 - k_p75_pro_on * aff_p75_pro * P * p75 + k_p75_pro_off * p75_pro - k_p75_B_on * aff_p75_B * B * p75 + k_p75_B_off * p75_B - k_degR1 * p75
        
        dTrkB = ks_TrkB - k_TrkB_B_on * aff_TrkB_B * B * TrkB + k_TrkB_B_off * TrkB_B - k_TrkB_pro_on * aff_TrkB_pro * P * TrkB + k_TrkB_pro_off * TrkB_pro - k_degR2 * TrkB
        
        dp75_pro = k_p75_pro_on * aff_p75_pro * P * p75 - k_p75_pro_off * p75_pro - k_int_p75_pro * p75_pro
        
        dp75_B = k_p75_B_on * aff_p75_B * B * p75 - k_p75_B_off * p75_B - k_int_p75_B * p75_B
        
        dTrkB_B = k_TrkB_B_on * aff_TrkB_B * B * TrkB - k_TrkB_B_off * TrkB_B - k_int_TrkB_B * TrkB_B
        
        dTrkB_pro = k_TrkB_pro_on * aff_TrkB_pro * P * TrkB - k_TrkB_pro_off * TrkB_pro - k_int_TrkB_pro * TrkB_pro
        
        dtPA = ks_tPA * activity_level - k_deg_tPA * tPA 



        return [dP, dB, dp75, dTrkB, dp75_pro, dp75_B, dTrkB_B, dTrkB_pro, dtPA]
    
    def solve(self, method, num_steps=1000):
        if method == 'manual':
            t0, tf = self.t_span
            dt = (tf - t0) / num_steps
            times = np.linspace(t0, tf, num_steps + 1)
            y = np.zeros((len(self.y0), num_steps + 1))
            y[:, 0] = self.y0
            self.growth_strength_history = []
            self.apop_strength_history = []

            for i in range(num_steps):
                current_y = y[:, i]
                dP, dB, dp75, dTrkB, dp75_pro, dp75_B, dTrkB_B, dTrkB_pro, dtPA = self.dYdt(times[i], current_y)
                dydt = [dP, dB, dp75, dTrkB, dp75_pro, dp75_B, dTrkB_B, dTrkB_pro, dtPA]
                y[:, i + 1] = current_y + np.array(dydt) * dt

                conc_TrkB_B = current_y[6] # concentrations
                conc_TrkB_pro = current_y[7] # concentrations
                conc_p75_pro = current_y[4] #concetrations
                conc_p75_B = current_y[5] # concentrations

                KD_TrkB_B = 0.05 # DISSACIATION CONSTANT FOR BDNF-TRKB COMPLEX (Change to actual)
                n_TrkB_B = 2.0    

                KD_TrkB_pro = 0.02 # DISSACIATION CONSTANT FOR PROBDNF-TRKB COMPLEX (Change to actual)
                n_TrkB_pro = 2.0 
                
                KD_p75_pro = 0.02 # DISSACIATION CONSTANT FOR P75-PROBDNF COMPLEX (Change to actual)
                n_p75_pro = 2.0 

                KD_p75_B = 0.02 # DISSACIATION CONSTANT FOR P75-BDNF COMPLEX (Change to actual)
                n_p75_B = 2.0

                growth_strength = (self.hill_function(conc_TrkB_B, KD_TrkB_B, n_TrkB_B) + 
                                     self.hill_function(conc_TrkB_pro, KD_TrkB_pro, n_TrkB_pro))/2
                
                apop_strength = (self.hill_function(conc_p75_pro, KD_p75_pro, n_p75_pro) +
                                        self.hill_function(conc_p75_B, KD_p75_B, n_p75_B))/2
                
                self.growth_strength_history.append(growth_strength)
                self.apop_strength_history.append(apop_strength)
            self.growth_strength_history = np.array(self.growth_strength_history)
            self.apop_strength_history = np.array(self.apop_strength_history)

            class Solution:
                pass
            sol = Solution()
            sol.t = times
            sol.y = y
            self.solution = sol
        elif method == 'scipy':
            sol = solve_ivp(self.dYdt, self.t_span, self.y0, method='RK45', t_eval=np.linspace(*self.t_span, num_steps))
            self.solution = sol

            KD_TrkB_B = 0.05
            n_TrkB_B = 2.0
            
            KD_p75_pro = 0.02
            n_p75_pro = 2.0

            self.growth_strength_history = np.array([
                self.hill_function(sol.y[6, i], KD=KD_TrkB_B, n=n_TrkB_B) for i in range(sol.y.shape[1])
            ])
            
            self.apop_strength_history = np.array([
                self.hill_function(sol.y[4, i], KD=KD_p75_pro, n=n_p75_pro) for i in range(sol.y.shape[1])
            ])

        return self.solution
        

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
    0.1     # tPA: tPA enzyme (moderate, present to allow cleavage)
]




# Parameters
params = [
    5.0e-3,   # ksP (synthesis rate of proBDNF)
    0.01,    # k_cleave (rate of proBDNF cleavage into BDNF)
    1.0,    # k_p75_pro_on (proBDNF binding to p75)
    0.9,    # k_p75_pro_off (proBDNF unbinding from p75)
    5.0e-4,   # k_degP (proBDNF degradation )
    0.2,    # k_TrkB_pro_on (proBDNF binding to TrkB)
    0.1,   # k_TrkB_pro_off (proBDNF unbinding from TrkB)
    1.0,    # k_TrkB_B_on (BDNF binding to TrkB)
    0.9,    # k_TrkB_B_off (BDNF unbinding from TrkB)
    0.005,    # k_degB (BDNF degradation)
    0.3,    # k_p75_B_on (BDNF binding to p75)
    0.2,   # k_p75_B_off (BDNF unbinding from p75)
    0.0001,   # k_degR1 (p75 degradation)
    0.00001,   # k_degR2 (TrkB degradation)
    0.0005,    # k_int_p75_pro (proBDNF-p75 internalization)
    0.0005,    # k_int_p75_B (BDNF-p75 internalization)
    0.0005,    # k_int_TrkB_B (BDNF-TrkB internalization)
    0.0005,    # k_int_TrkB_pro (proBDNF-TrkB internalization)
    0.9,    # aff_p75_pro (affinity of proBDNF for p75)
    0.1,    # aff_p75_B (affinity of BDNF for p75)
    0.1,    # aff_TrkB_pro (affinity of proBDNF for TrkB)
    0.9,    # aff_TrkB_B (affinity of BDNF for TrkB)
    0.0011,   #k_deg_tPA (degradation rate of tPA) - slow degradation
    0.0001,    # ks_tPA (synthesis rate of tPA)
    # NEW PARAMETERS FOR BIOLOGICAL ACCURACY
    0.0001,    # ks_p75 (synthesis rate of p75) - small value to maintain baseline
    0.00001,    # ks_TrkB (synthesis rate of TrkB) - small value to maintain baseline

]

t_span = (0, 2000)
ODE_model = ODEModel(params, y0, t_span)

t_eval = np.linspace(*t_span, 1000)
solution = ODE_model.solve(method='scipy', num_steps=1000)


x = []
y = []
for idx, t in enumerate(solution.t):
    derivatives = ODE_model.dYdt(t, solution.y[:, idx])
    x.append(solution.y[1][idx])  # dP/dt
    y.append(derivatives[1])  # dB/dt

plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=solution.t, cmap='viridis', label='Derivative Trajectory')
plt.xlabel('BdNF')
plt.ylabel('dB/dt (BDNF)')
plt.title('Transition of Derivatives Over Time')
plt.colorbar(label='Time')
plt.legend()
plt.grid()
plt.show()

x = []
y = []
for idx, t in enumerate(solution.t):
    derivatives = ODE_model.dYdt(t, solution.y[:, idx])
    x.append(solution.y[0][idx])  # dP/dt
    y.append(derivatives[0])  # dB/dt

plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=solution.t, cmap='viridis', label='Derivative Trajectory')
plt.xlabel('pBdNF')
plt.ylabel('dP/dt (BDNF)')
plt.title('Transition of Derivatives Over Time')
plt.colorbar(label='Time')
plt.legend()
plt.grid()
plt.show()

x = []
y = []
for idx, t in enumerate(solution.t):
    derivatives = ODE_model.dYdt(t, solution.y[:, idx])
    x.append(derivatives[0])  # dP/dt
    y.append(derivatives[1])  # dB/dt

plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=solution.t, cmap='viridis', label='Derivative Trajectory')
plt.xlabel('dP/dt')
plt.ylabel('dB/dt (BDNF)')
plt.title('Transition of Derivatives Over Time')
plt.colorbar(label='Time')
plt.legend()
plt.grid()
plt.show()
# Plot
# P, B, p75, TrkB, p75_pro, p75_B, TrkB_B, TrkB_pro, tPA
plt.figure(figsize=(10,6))
colors = plt.cm.tab10(np.arange(9))  # 9 variables, tab10 colormap
for i, label in enumerate(['proBDNF', 'BDNF', 'p75', 'TrkB', 'p75-proBDNF', 'p75-BDNF', 'TrkB_BDNF', 'TrkB_proBDNF', 'tPA']):
    plt.plot(solution.t, solution.y[i], label=label, color=colors[i])
    # Find local maxima and minima
    from scipy.signal import argrelextrema
    y = solution.y[i]
    t = solution.t
    max_idx = argrelextrema(y, np.greater)[0]
    min_idx = argrelextrema(y, np.less)[0]
    #plt.plot(t[max_idx], y[max_idx], '*', color=colors[i], markersize=8)  # Stars for maxima
    #plt.plot(t[min_idx], y[min_idx], '*', color=colors[i], markersize=8)  # Stars for minima
plt.legend()
plt.xlabel('Time: seconds')
plt.ylabel('Concentration: micromolar')
plt.title('Dynamics of proBDNF/BDNF and Receptors')
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
colors = plt.cm.tab10(np.arange(3))  # 2 variables
for i, label in enumerate(['proBDNF', 'BDNF', 'TPA']):
    plt.plot(solution.t, solution.y[i], label=label, color=colors[i])
    # Find local maxima and minima
    from scipy.signal import argrelextrema
    y = solution.y[i]
    t = solution.t
    max_idx = argrelextrema(y, np.greater)[0]
    min_idx = argrelextrema(y, np.less)[0]
    #plt.plot(t[max_idx], y[max_idx], '*', color=colors[i], markersize=8)
    #plt.plot(t[min_idx], y[min_idx], '*', color=colors[i], markersize=8)
plt.legend()
plt.xlabel('Time: seconds')
plt.ylabel('Concentration: micromolar')
plt.title('Dynamics of proBDNF/BDNF and Receptors and TPA')
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
colors = plt.cm.tab10(np.arange(3))  # 9 variables, tab10 colormap
for i, label in enumerate(['p75', 'p75-proBDNF', 'p75-BDNF']):
    plt.plot(solution.t, solution.y[i], label=label, color=colors[i])
    # Find local maxima and minima
    from scipy.signal import argrelextrema
    y = solution.y[i]
    t = solution.t
    max_idx = argrelextrema(y, np.greater)[0]
    min_idx = argrelextrema(y, np.less)[0]
    #plt.plot(t[max_idx], y[max_idx], '*', color=colors[i], markersize=8)  # Stars for maxima
    #plt.plot(t[min_idx], y[min_idx], '*', color=colors[i], markersize=8)  # Stars for minima
plt.legend()
plt.xlabel('Time: seconds')
plt.ylabel('Concentration: micromolar')
plt.title('Dynamics of p75 binding')
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
colors = plt.cm.tab10(np.arange(4))  # 9 variables, tab10 colormap
for i, label in enumerate(['TrkB', 'TrkB_BDNF', 'TrkB_proBDNF', 'tPA']):
    plt.plot(solution.t, solution.y[i], label=label, color=colors[i])
    # Find local maxima and minima
    from scipy.signal import argrelextrema
    y = solution.y[i]
    t = solution.t
    max_idx = argrelextrema(y, np.greater)[0]
    min_idx = argrelextrema(y, np.less)[0]
    #plt.plot(t[max_idx], y[max_idx], '*', color=colors[i], markersize=8)  # Stars for maxima
    #plt.plot(t[min_idx], y[min_idx], '*', color=colors[i], markersize=8)  # Stars for minima
plt.legend()
plt.xlabel('Time: seconds')
plt.ylabel('Concentration: micromolar')
plt.title('trkb and all trkb stuff')
plt.grid()
plt.show()

# --- New Plots for Growth and Apoptosis Strengths ---
plt.figure(figsize=(10,6))
plt.plot(solution.t, ODE_model.growth_strength_history-ODE_model.apop_strength_history, label='Growth Signal', color='purple')
plt.xlabel('Time: seconds')
plt.ylabel('Signal Strength (0-1)')
plt.title('Overall Growth Signal')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(solution.t, ODE_model.apop_strength_history, label='Apoptosis Signal', color='darkred')
plt.xlabel('Time: seconds')
plt.ylabel('Signal Strength (0-1)')
plt.title('Overall Apoptosis Signal')
plt.grid()
plt.legend()
plt.show()