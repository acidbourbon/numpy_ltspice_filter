#!/usr/bin/env python3

import sys
import numpy as np
from apply_ltspice_filter import apply_ltspice_filter
import matplotlib.pyplot as plt

##################################################
##             generate test signal             ##
##################################################

# our samples shall be 100 ms wide
sample_width=100e-3
# time step between samples: 0.1 ms
delta_t=0.1e-3
samples = int(sample_width/delta_t)

time = np.linspace(0,sample_width,samples)

# we want 1 V between 10 ms and 30 ms, and 2.5 V between 40 and 70 ms
signal_a = 0 + 1*((time > 10e-3) * (time < 30e-3)) + 2.5*((time > 40e-3) * (time < 70e-3))



if sys.platform == "darwin":
  """ In order for the command /Applications/LTspice.app/Contents/MacOS/LTspice -b Draft1.cir to work, a netlist file is required."""
  file_extension = "cir"
else:
  file_extension = "asc"



##################################################
##        apply filter - configuration 1        ##
##################################################

# all values in SI units
configuration_1 = {
  "C":100e-6, # 100 uF
  "L":200e-3 # 200 mH
}

dummy, signal_b1 = apply_ltspice_filter(
      f"filter_circuit.{file_extension}",
      time, signal_a,
      params=configuration_1
      )

##################################################
##        apply filter - configuration 2        ##
##################################################

configuration_2 = {
  "C":50e-6, # 50 uF
  "L":300e-3 # 300 mH
}

dummy, signal_b2 = apply_ltspice_filter(
      f"filter_circuit.{file_extension}",
      time, signal_a,
      params=configuration_2
      )

##################################################
##           plot input vs output(s)            ##
##################################################
  
plt.plot(time,signal_a, label="signal_a (LTSpice input)")
plt.plot(time,signal_b1, label="signal_b1 (LTSpice output, C=100uF, L=200mH)")
plt.plot(time,signal_b2, label="signal_b2 (LTSpice output, C=50uF,  L=300mH)")
plt.xlabel("time (s)")
plt.ylabel("voltage (V)")
plt.ylim((-1,3.5))
plt.grid(True)

plt.legend()
plt.show()
