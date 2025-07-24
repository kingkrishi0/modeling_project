import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

""""
P(t) = proBDNF
B(t) = BDNF
R1(t) = free p75
R2(t) = free TrkB
C1(t) = proBDNF-p75 complex
C2(t) = BDNF-TrkB complex
ksP = synthesis rate of proBDNF moles/time
k_cleave = rate of proBDNF cleavage into BDNF (frequency) 1/time
k_on1 = rate of proBDNF to bind to p75 1/(moles*time)
k_off1 = rate of proBDNF to unbind from the proBDNF-p75 1/time
k_degP = rate of proBDNF degradation 1/time
k_on2 = rate of BDNF to bind to TrkB 1/(moles*time)
k_off2 = rate of BDNF to unbind from the BDNF-TrkB complex 1/time
k_degB = rate of BDNF degradation 1/time
k_degR1 = rate of p75 degradation 1/time
k_degR2 = rate of TrkB degradation 1/time
k_int1 = rate of proBDNF-p75 complex internalization 1/time
k_int2 = rate of BDNF-TrkB complex internalization 1/time
"""

# Define the ODE system
def dYdt(t, y, params):
    P, B, R1, R2, C1, C2 = y
    ksP, k_cleave, k_on1, k_off1, k_degP, \
    k_on2, k_off2, k_degB, k_degR1, k_degR2, \
    k_int1, k_int2 = params

    dP = ksP - k_cleave*P - k_on1*P*R1 + k_off1*C1 - k_degP*P
    dB = k_cleave*P - k_on2*B*R2 + k_off2*C2 - k_degB*B
    dR1 = -k_on1*P*R1 + k_off1*C1 - k_degR1*R1
    dR2 = -k_on2*B*R2 + k_off2*C2 - k_degR2*R2
    dC1 = k_on1*P*R1 - k_off1*C1 - k_int1*C1
    dC2 = k_on2*B*R2 - k_off2*C2 - k_int2*C2

    return [dP, dB, dR1, dR2, dC1, dC2]

# Initial concentrations
y0 = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]  # [P, B, R1, R2, C1, C2]

# Parameters
params = [
    1.0,    # ksP
    0.5,    # k_cleave
    1.0,    # k_on1
    0.1,    # k_off1
    0.05,   # k_degP
    1.0,    # k_on2
    0.1,    # k_off2
    0.5,   # k_degB
    0.01,   # k_degR1
    0.01,   # k_degR2
    0.1,    # k_int1
    0.1     # k_int2
]

# Time span
t_span = (0, 50)
t_eval = np.linspace(*t_span, 500)

# Solve
solution = solve_ivp(dYdt, t_span, y0, args=(params,), t_eval=t_eval)

# Plot
plt.figure(figsize=(10,6))
for i, label in enumerate(['proBDNF', 'BDNF', 'p75', 'TrkB', 'C1 (proBDNF-p75)', 'C2 (BDNF-TrkB)']):
    plt.plot(solution.t, solution.y[i], label=label)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Dynamics of proBDNF/BDNF and Receptors')
plt.grid()
plt.show()