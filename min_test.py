from neuron import h
import os

os.system("nrnivmodl")
h.nrn_load_dll("x86_64/.libs/libnrnmech.so")

soma = h.Section()
soma.L = 20
soma.diam = 20
soma.cm = 1.0
soma.insert('pas')  # Use built-in passive for membrane
soma.g_pas = 0.0001
soma.e_pas = -65
soma.insert('ode_neuron')  # Add your mechanism

stim = h.IClamp(soma(0.5))
stim.delay = 10
stim.dur = 50
stim.amp = 0.1  # ← Use the SAME reduced amplitude

h.finitialize(-65)
h.dt = 0.025

for i in range(1000):
    h.fadvance()
    v = soma(0.5).v
    if abs(v) > 100:
        print(f"With mechanism exploded: {v:.1f} mV")
        break
    if i % 100 == 0:
        print(f"t={h.t:.1f}, V={v:.2f}")