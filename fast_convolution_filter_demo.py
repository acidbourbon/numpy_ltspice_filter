#!/usr/bin/env python3

import numpy as np
from apply_ltspice_filter import \
  apply_ltspice_filter,\
  convolution_filter,\
  get_impulse_response
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

##################################################
##             apply filter direct              ##
##################################################

# all values in SI units
filter_configuration = {
  "C":100e-6, # 100 uF
  "L":200e-3  # 200 mH
}

dummy, signal_b = apply_ltspice_filter(
      "filter_circuit.asc",
      time, signal_a,
      params = filter_configuration
      )

##################################################
##                     plot                     ##
##################################################

plt.plot(time,signal_a, label="signal_a (LTSpice input)")
plt.plot(time,signal_b, label="signal_b (LTSpice output)")
plt.xlabel("time (s)")
plt.ylabel("voltage (V)")
plt.title("direct processing")
plt.ylim((-1,3.5))

plt.legend()
plt.show()

##################################################
##             get impulse response             ##
##################################################

kernel_delay = 10e-3
kernel_sample_width = 100e-3

kernel_time, kernel = get_impulse_response(
        "filter_circuit.asc",
        params = filter_configuration,
        sample_width = kernel_sample_width,
        delta_t = delta_t,
        kernel_delay = kernel_delay
        )

##################################################
##            plot impulse response             ##
##################################################

plt.plot(kernel_time, kernel, label="impulse response of filter_circuit.asc")
plt.xlabel("time (s)")
plt.ylabel("voltage (V)")
plt.title("impulse response")

plt.legend()
plt.show()

##################################################
##           apply convolution filter           ##
##################################################

signal_b_conv = convolution_filter(
  signal_a,
  kernel,
  delta_t = delta_t,
  kernel_delay = kernel_delay
  )



##################################################
##      plot direct vs convolution filter       ##
##################################################
  
plt.plot(time,signal_a, label="signal_a (LTSpice input)")
plt.plot(time,signal_b, label="signal_b (LTSpice output)")
plt.plot(time,signal_b_conv, label="signal_b via convolution with IR", linestyle="-.")
plt.xlabel("time (s)")
plt.ylabel("voltage (V)")
plt.ylim((-1,3.5))
plt.title("direct processing vs convolution filter")

plt.legend()
plt.show()


##################################################
##        demonstration of speed advantage      ##
##################################################

for i in range(0,10):
  ts = i * 6e-3
  signal = 0 + (5+i)*(((time-ts) > 10e-3) * ((time-ts) < (10+2*i)*1e-3)) 
  if i == 0:
    plt.plot(time,signal,color="b",label="in",alpha= 0.9)
  else:
    plt.plot(time,signal,color="b",alpha= 0.9)
  
  signal_conv = convolution_filter(
    signal,
    kernel,
    delta_t = delta_t,
    kernel_delay = kernel_delay
    )
  
  if i == 0:
    plt.plot(time,signal_conv,color="orange",label="out",alpha=0.9)
  else:
    plt.plot(time,signal_conv,color="orange",alpha=0.9)
  
plt.xlabel("time (s)")
plt.ylabel("voltage (V)")
plt.title("demonstration of convolution speed advantage")
plt.ylim((-5,20))
plt.legend()
plt.show()
  




