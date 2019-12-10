#!/usr/bin/env python3

import numpy as np
import os
from scipy import interpolate, signal
import filecmp
from shutil import copyfile
import sys

# use Nuno Brum's RawRead module. Can be found here:
# https://github.com/nunobrum/PyLTSpice/raw/master/LTSpice_RawRead.py
#from LTSpice_RawRead import RawRead

# use Nuno's PyPi module
from PyLTSpice.LTSpice_RawRead import RawRead



def gauss(x, **kwargs):
  mu = kwargs.get("mu",0)
  sigma = kwargs.get("sigma",1)
  A = kwargs.get("A",1./(sigma*(2.*np.pi)**0.5)) ## default amplitude generates bell curve with area = 1
  return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def resize_vector( vector, target_size):
    if ( len(vector) < target_size ):
      return np.pad( vector,(0,target_size-len(vector)), 'constant', constant_values=(0) ) ## pad with zeros to desired length
    elif ( len(vector) > target_size ):
      return vector[0:target_size] ## cut away if too long
    else:
      return vector


def resample(target_x,data_x,data_y):
  f = interpolate.interp1d(data_x,data_y,bounds_error=False, fill_value=0.)
  out_x = target_x
  out_y = f(target_x)
  return (out_x,out_y)


def convolution_filter(data, kernel, **kwargs):
  
  delta_t = float(kwargs.get("delta_t",1))
  kernel_delay = float(kwargs.get("kernel_delay",0))
  
  # apply filter kernel to input data via fft convolution
  filtered = signal.fftconvolve( data , kernel * delta_t , mode = 'full' )

  # shift filtered signal backwards in time to counteract kernel_delay
  filtered = filtered[ int(kernel_delay/delta_t): ]

  # bring filtered signal to same length as input data, cut away at the end
  filtered = resize_vector(filtered,len(data))
  
  return filtered



def get_impulse_response(simname, **kwargs):
  params  = kwargs.get("params",{})
  
  
  spice_sample_width= float(kwargs.get("sample_width",1))

  delta_t = float( kwargs.get("delta_t",1))
  
  spice_delta_t = float( kwargs.get("spice_delta_t", delta_t/4. ))
  
  spice_samples = int(spice_sample_width/spice_delta_t)
  spice_time = np.linspace(0,spice_sample_width,spice_samples)
  
  kernel_delay = float(kwargs.get("kernel_delay", spice_sample_width*0.1)) # delta pulse at 10% of sample width
  
  delta_pulse = gauss( spice_time,
                     mu = kernel_delay,
                     sigma=2*spice_delta_t
                   )


  dummy, spice_IR = apply_ltspice_filter(
        simname,
        spice_time,
        delta_pulse,
        params = params
        )
  
  
  kernel_sample_width = spice_sample_width

  kernel_delta_t = delta_t
  kernel_samples = int(kernel_sample_width/kernel_delta_t)
  kernel_time = np.linspace(0,kernel_sample_width,kernel_samples)

  return resample(kernel_time, spice_time, spice_IR)
  # returns (kernel_time, kernel)
  

  

def apply_ltspice_filter(simname,sig_in_x,sig_in_y,**kwargs):
  
  verbose = kwargs.get("verbose",False)
  interpol = kwargs.get("interpolate",True)
  
  default_ltspice_command = "C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe -Run -b " 
  if sys.platform == "linux":
    default_ltspice_command = 'wine C:\\\\Program\\ Files\\\\LTC\\\\LTspiceXVII\\\\XVIIx64.exe -Run -b '
  
  ltspice_command = kwargs.get("ltspice_command",default_ltspice_command)
  
  

  params  = kwargs.get("params",{})

  simname = simname.replace(".asc","")

  with open("sig_in.csv_","w") as f:
    for i in range(0,len(sig_in_x)):
      f.write("{:E}\t{:E}\n".format(sig_in_x[i],sig_in_y[i]))
    f.close()
    
  with open("trancmd.txt_","w") as f:
    f.write(".param transtop {:E}\n".format(sig_in_x[-1]-sig_in_x[0]))
    f.write(".param transtart {:E}\n".format(sig_in_x[0]))
    f.write(".param timestep {:E}\n".format(sig_in_x[1]-sig_in_x[0]))
    f.close()

  with open("param.txt_","w") as f:
    for key in params:
      f.write(".param {:s} {:E}\n".format( key,params[key]  ))
    f.write("\n")
    f.close()
    

  sth_changed = False

  # check if we ran the simulation before with exact same input, can save time
  if os.path.isfile('sig_in.csv') and filecmp.cmp('sig_in.csv_', 'sig_in.csv') :
    print("sig_in.csv has not changed")
  else:
    sth_changed = True
    copyfile('sig_in.csv_', 'sig_in.csv')
    
  if os.path.isfile('trancmd.txt') and filecmp.cmp('trancmd.txt_', 'trancmd.txt'): 
    print("trancmd.txt has not changed")
  else:
    sth_changed = True
    copyfile('trancmd.txt_', 'trancmd.txt')
    
  if os.path.isfile('param.txt') and filecmp.cmp('param.txt_','param.txt') : 
    print("param.txt has not changed")
  else:
    sth_changed = True
    copyfile('param.txt_','param.txt')
    
    
  if os.path.isfile("{:s}.raw".format(simname)): ## raw file already exists
    # get rawfile modification date
    rawmdate = os.path.getmtime("{:s}.raw".format(simname))
    # get ascfile modification date
    ascmdate = os.path.getmtime("{:s}.asc".format(simname))
    if ascmdate > rawmdate: # asc file has been modified in the meantime
      print("{:s}.asc is newer than {:s}.raw".format(simname,simname))
      sth_changed = True
    else:
      print("{:s}.asc is older than {:s}.raw".format(simname,simname))
  else :
    sth_changed = True

  # do not execute ltspice if nothing has changed
  if sth_changed:
    #print("executing ./wine_ltspice.sh, saving STDOUT to wine_ltspice.log")
    #os.system("{:s} {:s}.asc > wine_ltspice.log 2>&1".format(simname))
    if sys.platform == "linux":
      os.system(ltspice_command+" {:s}.asc".format(simname))
    else:
      import subprocess
      subprocess.run([*ltspice_command.split(), "{:s}.asc".format(simname)])
    
  else:
    print("input data did not change, reading existing .raw file")
    
  ltr = RawRead("{:s}.raw".format(simname))
  
  
  if verbose:
    for name in ltr.get_trace_names():
      for step in ltr.get_steps():
        tr = ltr.get_trace(name)
        print(name)
        print('step {:d} {}'.format(step, tr.get_wave(step)))
  
  #os.system("./clean_up.sh")
  os.remove("param.txt_")
  os.remove("trancmd.txt_")
  os.remove("sig_in.csv_")
  
  IR1 = ltr.get_trace("V(vout)")
  x = ltr.get_trace("time") 
  
  #  #### the abs() is a quick and dirty fix for some strange sign decoding errors
  vout_x = abs(x.get_wave(0))
  vout_y = IR1.get_wave(0)
 
  #  interpolate ltspice output, so you have the same x value spacing as in the input voltage vector
  if interpol:
    f = interpolate.interp1d(vout_x,vout_y)
    vout_x = sig_in_x
    vout_y = f(sig_in_x)
  
  return (vout_x,vout_y)









