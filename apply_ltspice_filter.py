#!/usr/bin/env python3

import numpy as np
#from LTSpice_RawRead import RawRead  ## i attached this module underneath for convenience
import os
from scipy import interpolate, signal

import filecmp
from shutil import copyfile
import sys




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



















################################################################################################

## the following code is module LTSpice_RawRead by Nuno Brum ##

## i did not write this ##
################################################################################################

































#-------------------------------------------------------------------------------
# Name:        LTSpice_RawRead.py
# Purpose:     Process LTSpice output files and align data for usage in a spread-
#              sheet tool such as Excel, or Calc.
#
# Author:      Nuno Brum (nuno.brum@gmail.com)
#
# Created:     23-12-2016
# Licence:     General Public GNU License
#-------------------------------------------------------------------------------

""" A pure python implementation of an LTSpice RAW file reader.
The reader returns a class containing all the traces read from the RAW File.
In case there there stepped data detected, it will try to open the simulation LOG file and
read the stepping information.
Traces are accessible by the method <LTSpiceReader instance>.get_trace(trace_ref) where trace_ref is either
the name of the net on the LTSPice Simulation. Normally trace references are stored with the format V(<node_name>)
for voltages or I(device_reference). For example V(n001) or I(R1) or Ib(Q1).
For checking step, the method <LTSpiceReader instance>.get_steps() is used. In case there are no steps in the simulation,
the class will return a single element list.
NOTE: This module tries to import the numpy if exists on the system.
If it finds numpy all data is later provided as an array. If not it will use a standard list of floats.
"""

__author__ = "Nuno Canto Brum <nuno.brum@gmail.com>"
__copyright__ = "Copyright 2017, Fribourg Switzerland"

import os
from binascii import b2a_hex
from struct import unpack
try:
    from numpy import zeros, array, complex128
except ImportError:
    USE_NNUMPY = False
else:
    USE_NNUMPY = True
    print("Found Numpy. WIll be used for storing data")


class DataSet(object):
    """Class for storing Traces."""

    def __init__(self, name, datatype, datalen, numerical_type='real'):
        """Base Class for both Axis and Trace Classes.
        Defines the common operations between both."""
        self.name = name
        self.type = datatype
        self.numerical_type = numerical_type
        if USE_NNUMPY:
            if numerical_type == 'real':
                self.data = zeros(datalen)
            elif numerical_type == 'complex':
                self.data = zeros(datalen, complex128)
        else:
            self.data = [None for x in range(datalen)]

    def set_pointA(self, n, value):
        """function to be used on ASCII RAW Files.
        :param n:     the point to set
        :param value: the Value of the point being set."""
        assert isinstance(value, float)
        self.data[n] = value

    def set_pointB8(self, n, value):
        """Function that converts the variable 0, normally associated with the plot X axis.
        The codification is done as follows:
               7   6   5   4     3   2   1   0
        Byte7  SGM SGE E9  E8    E7  E6  E5  E4         SGM - Signal of Mantissa: 0 - Positive 1 - Negative
        Byte6  E3  E2  E1  E0    M51 M50 M49 M48        SGE - Signal of Exponent: 0 - Positive 1 - Negative
        Byte5  M47 M46 M45 M44   M43 M42 M41 M40        E[9:0] - Exponent
        Byte4  M39 M38 M37 M36   M35 M34 M33 M32        M[51:0] - Mantissa.
        Byte3  M31 M30 M29 M28   M27 M26 M25 M24
        Byte2  M23 M22 M21 M20   M19 M18 M17 M16
        Byte1  M15 M14 M13 M12   M11 M10 M9  M8
        Byte0  M7  M6  M5  M4    M3  M2  M1  M0
        """
        self.data[n] = unpack("d", value)[0]

    def set_pointB16(self, n, value):
        (re, im) = unpack('dd', value)
        self.data[n] = complex(re, im)

    def set_pointB4(self, n, value):
        """Function that converts a normal trace into float on a Binary storage. This codification uses 4 bytes.
        The codification is done as follows:
               7   6   5   4     3   2   1   0
        Byte3  SGM SGE E6  E5    E4  E3  E2  E1         SGM - Signal of Mantissa: 0 - Positive 1 - Negative
        Byte2  E0  M22 M21 M20   M19 M18 M17 M16        SGE - Signal of Exponent: 0 - Positive 1 - Negative
        Byte1  M15 M14 M13 M12   M11 M10 M9  M8         E[6:0] - Exponent
        Byte0  M7  M6  M5  M4    M3  M2  M1  M0         M[22:0] - Mantissa.
        """
        self.data[n] = unpack("f", value)[0]

    def __str__(self):
        if isinstance(self.data[0], float):
            # data = ["%e" % value for value in self.data]
            return "name:'%s'\ntype:'%s'\nlen:%d\n%s" % (self.name, self.type, len(self.data), str(self.data))
        elif isinstance(self.data[0], complex):
            return "name: {}\ntype: {}\nlen: {:d}\n{}".format(self.name, self.type, len(self.data), str(self.data))
        else:
            data = [b2a_hex(value) for value in self.data]
            return "name:'%s'\ntype:'%s'\nlen:%d\n%s" % (self.name, self.type, len(self.data), str(data))

    def get_point(self, n):
        return self.data[n]

    def get_wave(self):
        return self.data

    def get_len(self):
        return len(self.data)


class Axis(DataSet):
    """This class is used to represent the horizontal axis like on a Transient or DC Sweep Simulation."""

    def __init__(self, name, datatype, datalen, numerical_type='real'):
        super().__init__(name, datatype, datalen, numerical_type)
        self.step_info = None

    def _set_steps(self, step_info):
        self.step_info = step_info

        self.step_offsets = [None for x in range(len(step_info))]

        # Now going to calculate the point offset for each step
        self.step_offsets[0] = 0
        i = 0
        k = 0
        while i < len(self.data):
            if self.data[i] == self.data[0]:
                # print(k, i, self.data[i], self.data[i+1])
                if self.data[i] == self.data[i+1]:
                    i += 1  # Needs to add one here because the data will be repeated
                self.step_offsets[k] = i
                k += 1
            i += 1

        if k != len(self.step_info):
            raise LTSPiceReadException("The file a different number of steps than expected.\n" +
                                       "Expecting %d got %d" % (len(self.step_offsets), k))

    def step_offset(self, step):
        if self.step_info is None:
            if step > 0:
                return len(self.data)
            else:
                return 0
        else:
            if step >= len(self.step_offsets):
                return len(self.data)
            else:
                return self.step_offsets[step]

    def get_wave(self, step=0):
        return self.data[self.step_offset(step):self.step_offset(step + 1)]


class Trace(DataSet):
    """Class used for storing generic traces that report to a given Axis."""

    def __init__(self, name, datatype, datalen, axis, numerical_type='real'):
        super().__init__(name, datatype, datalen, numerical_type)
        self.axis = axis

    def get_point(self, n=0, step=0):
        if self.axis is None:
            return super().get_point(n)
        else:
            return self.data[self.axis.step_offset(step) + n]

    def get_wave(self, step=0):
        if self.axis is None:
            return super().get_wave()
        else:
            return self.data[self.axis.step_offset(step):self.axis.step_offset(step + 1)]


class Op(Trace):
    """Class used for storing operation points."""
    pass


class DummyTrace(object):
    """Dummy Trace for bypassing traces while reading"""

    def __init__(self, name, datatype):
        """Base Class for both Axis and Trace Classes.
        Defines the common operations between both."""
        self.name = name
        self.type = datatype

    def set_pointA(self, n, value):
        pass

    def set_pointB8(self, n, value):
        pass

    def set_pointB4(self, n, value):
        pass


class LTSPiceReadException(Exception):
    """Custom class for exception handling"""


class LTSpiceRawRead(object):
    """Class for reading LTSpice wave Files. It can read all types of Files. If stepped data is detected,
    it will also try to read the corresponding LOG file so to retrieve the stepped data.
    """
    header_lines = [
        "Title",
        "Date",
        "Plotname",
        "Output",
        "Flags",
        "No. Variables",
        "No. Points",
        "Offset",
        "Command",
        "Variables",
        "Backannotation"
    ]

    def __init__(self, raw_filename, traces_to_read="*", **kwargs):
        """The arguments for this class are:
    raw_filename   - The file containing the RAW data to be read
    traces_to_read - A string containing the list of traces to be read. If None is provided, only the header is read
                     and all trace data is discarded. If a '*' wildcard is given, all traces are read.
    kwargs         - Keyword parameters that define the options for the loading. Options are:
                        loadmem - If true, the file will only read waveforms to memory
    """
        assert isinstance(raw_filename, str)
        if not traces_to_read is None:
            assert isinstance(traces_to_read, str)

        self.encoding = 'utf_16_le'
        self.offset = 1

        raw_file_size = os.stat(raw_filename).st_size # Get the file size in order to know the data size
        raw_file = open(raw_filename, "rb")

        # Storing the filename as part of the dictionary
        self.raw_params = {"Filename": raw_filename}  # Initializing the dictionary that contains all raw file info
        self.backannotations = []  # Storing backannotations
        
        startpos = 0  # counter of bytes for

        # LTSpice raw_files are encoded in UTF-16-le. We ignore errors because
        # readline stops reading lines after the '\n' and doesn't include 0x00
        # that occurs after
        line = raw_file.readline().decode(encoding=self.encoding, errors='ignore')
        raw_file.read(self.offset)  # correct issue with utf16 decoding

        # check for correct encoding
        # TODO: do this better
        if not line.startswith('Title'):
            print("Ascii file")
            self.encoding = 'utf_8'
            self.offset = 0
            raw_file.seek(0)
            line = raw_file.readline().decode(encoding=self.encoding, errors='ignore')
        else:
            print("Binary file")

        while line:
            # print(line)
            startpos += len(line)
            tag, value_str = line.split(':', 1)
            if tag in self.header_lines:
                if tag == 'Backannotation':
                    self.backannotations.append(value_str)
                else:
                    self.raw_params[tag] = value_str
            else:
                raw_file.close()
                raise LTSPiceReadException(("Error reading Raw File !\n " +
                                            "Unrecognized tag in line %s") % line)
            line = raw_file.readline().decode(encoding=self.encoding, errors='ignore')
            raw_file.read(self.offset)  # correct issue with utf16 decoding
            if line.startswith("Variables:"):
                break
        else:
            raw_file.close()
            raise LTSPiceReadException("Error reading Raw File !\n " +
                                       "Unexpected end of file")

        if not ("real" in self.raw_params["Flags"]):
            # Not Supported, an exception will be raised
            raw_file.close()
            raise LTSPiceReadException("The LTSpiceRead class doesn't support non real data")

        self.nPoints = int(self.raw_params["No. Points"], 10)
        self.nVariables = int(self.raw_params["No. Variables"], 10)
        self._traces = []
        self.steps = None
        self.axis = None  # Creating the axis
        # print("Reading Variables")

        ivar = 0
        while line:
            line = raw_file.readline()\
                    .decode(encoding=self.encoding, errors='ignore')
            raw_file.read(self.offset)
            print(line)

            if line.startswith("Binary:") or line.startswith("Values"):
                if ivar != self.nVariables:
                    raw_file.close()
                    raise LTSPiceReadException("Wrong number of variables read")
                self.raw_type = line
                self.binary_start = raw_file.tell()
                break

            dummy, _, name, var_type = line.split("\t")
            if ivar == 0 and self.nVariables > 1 and self.nPoints != 1:
                self.axis = Axis(name, var_type, self.nPoints)
                self._traces.append(self.axis)
            elif self.nPoints == 1:
                self._traces.append(Op(name, var_type, self.nPoints, self.axis))
            elif ((traces_to_read == "*") or
                      (name in traces_to_read) or
                      (ivar == 0)):
                # TODO: Add wildcards to the waveform matching
                self._traces.append(Trace(name, var_type, self.nPoints, self.axis))
            else:
                self._traces.append(DummyTrace(name, var_type))
            ivar += 1

        if traces_to_read is None or len(self._traces) == 0:
            # The read is stopped here if there is nothing to read.
            raw_file.close()
            return

        # This will make a lazy loading. That means, only the Axis is read. The traces are only read when the user
        # makes a get_trace()
        self.in_memory = False  # point to set it to true at the end of the load

        if kwargs.get("headeronly", False):
            raw_file.close()
            return

        if self.raw_type.startswith("Binary:"):
            # Will start the reading of binary values
            # But first check whether how data is stored.
            self.block_size = (raw_file_size - self.binary_start) // self.nPoints
            self.data_size = (self.block_size - 8) // (self.nVariables - 1)
            if "fastaccess" in self.raw_params["Flags"]:
                print("Fast access")
                # A fast access means that the traces are grouped together.
                for var in self._traces:
                    if isinstance(var, DummyTrace):
                        # TODO: replace this by a seek
                        raw_file.read(self.nPoints * self.data_size)
                    else:
                        if self.data_size == 8 or isinstance(var, Axis):
                            for point in range(self.nPoints):
                                value = raw_file.read(8)
                                var.set_pointB8(point, value)
                        else:  # Data size is 4
                            for point in range(self.nPoints):
                                value = raw_file.read(4)
                                var.set_pointB4(point, value)

            else:
                print("Normal access")
                # This is the default save after a simulation where the traces are scattered
                if self.data_size == 8:
                    for point in range(self.nPoints):
                        for var in self._traces:
                            value = raw_file.read(8)
                            var.set_pointB8(point, value)
                else:  # data size is only 4 bytes
                    for point in range(self.nPoints):
                        value = raw_file.read(8)  # first variable (ex:time) is always 8 bytes
                        self._traces[0].set_pointB8(point, value)

                        for var in self._traces[1:]:
                            value = raw_file.read(4)
                            var.set_pointB4(point, value)

        elif self.raw_type.startswith("Values:"):
            # Will start the reading of ASCII Values
            for point in range(self.nPoints):
                first_var = True
                for var in self._traces:
                    line = raw_file.readline()\
                            .decode(encoding=self.encoding, errors='ignore')
                    raw_file.seek(raw_file.tell() + self.offset) # Move past 0x00 from prev. line
                    # print(line)

                    if first_var:
                        first_var = False
                        spoint = line.split("\t", 1)[0]
                        # print(spoint)
                        if point != int(spoint):
                            print("Error Reading File")
                            break
                        value = float(line[len(spoint):-1])
                    else:
                        value = float(line[:-1])
                    var.set_pointA(point, value)
        else:
            raw_file.close()
            raise LTSPiceReadException("Unsupported RAW File. ""%s""" % self.raw_type)

        raw_file.close()

        # Setting the properties in the proper format
        self.raw_params["No. Points"] = self.nPoints
        self.raw_params["No. Variables"] = self.nVariables
        self.raw_params["Variables"] = [var.name for var in self._traces]

        # Now Purging Dummy Traces
        i = 0
        while i < len(self._traces):
            if isinstance(self._traces[i], DummyTrace):
                del self._traces[i]
            else:
                i += 1

        # Finally, Check for Step Information
        if "stepped" in self.raw_params["Flags"]:
            self._load_step_information(raw_filename)

    def get_raw_property(self, property_name=None):
        """Get a property. By default it returns everything"""
        if property_name is None:
            return self.raw_params
        elif property_name in self.raw_params.keys():
            return self.raw_params[property_name]
        else:
            return "Invalid property. Use %s" % str(self.raw_params.keys())

    def get_trace_names(self):
        return [trace.name for trace in self._traces]

    def get_trace(self, trace_ref):
        """Retrieves the trace with the name given. """
        if isinstance(trace_ref, str):
            for trace in self._traces:
                if trace_ref == trace.name:
                    # assert isinstance(trace, DataSet)
                    return trace
            return None
        else:
            return self._traces[trace_ref]

    def _load_step_information(self, filename):
        # Find the extension of the file
        if not filename.endswith(".raw"):
            raise LTSPiceReadException("Invalid Filename. The file should end with '.raw'")
        logfile = filename[:-3] + 'log'
        try:
            log = open(logfile, 'r')
        except:
            raise LTSPiceReadException("Step information needs the '.log' file generated by LTSpice")

        for line in log:
            if line.startswith(".step"):
                step_dict = {}
                for tok in line[6:-1].split(' '):
                    key, value = tok.split('=')
                    try:
                        # Tries to convert to float for backward compatibility
                        value = float(value)
                    except:
                        pass
                        # Leave value as a string to accomodate cases as temperature steps
                        # Temperature steps have the form '.step temp=25°C'
                    step_dict[key] = value

                if self.steps is None:
                    self.steps = [step_dict]
                else:
                    self.steps.append(step_dict)
        log.close()
        if not (self.steps is None):
            # Individual access to the Trace Classes, this information is stored in the Axis
            # which is always in position 0
            self._traces[0]._set_steps(self.steps)
            pass

    def __getitem__(self, item):
        """Helper function to access traces by using the [ ] operator."""
        return self.get_trace(item)

    def get_steps(self, **kwargs):
        if self.steps is None:
            return [0]  # returns an single step
        else:
            if len(kwargs) > 0:
                ret_steps = []  # Initializing an empty array
                i = 0
                for step_dict in self.steps:
                    for key in kwargs:
                        ll = step_dict.get(key, None)
                        if ll is None:
                            break
                        elif kwargs[key] != ll:
                            break
                    else:
                        ret_steps.append(i)  # All the step parameters match
                    i += 1
                return ret_steps
            else:
                return range(len(self.steps))  # Returns all the steps


class RawRead(object):

    header_lines = [
        "Title",
        "Date",
        "Plotname",
        "Flags",
        "No. Variables",
        "No. Points",
        "Offset",
        "Command",
        "Variables",
        "Backannotation"
    ]
    
    def __init__(self, raw_filename, traces_to_read='*', **kwargs):
        self.raw_params = {'Filename': raw_filename}
        raw_file_size = os.stat(raw_filename).st_size
        self.backannotations = []
        raw_file = open(raw_filename, 'rb')
        header = []
        while True:
            ch = raw_file.read(2).decode(encoding='utf_16_le')
            header.append(ch)
            if header[-8:] == ['B', 'i', 'n', 'a', 'r', 'y', ':', '\n']:
                self.binary_start = raw_file.tell()
                break
        header = ''.join(header).split('\n')[:-2]
        for line in header:
            k, _, v = line.partition(':')
            if k == 'Variables':
                break
            self.raw_params[k] = v
        self.nPoints = int(self.raw_params['No. Points'])
        self.nVariables = int(self.raw_params['No. Variables'])
        self._traces = []
        self.steps = None
        self.axis = None
        if 'complex' in self.raw_params['Flags']:
            numerical_type = 'complex'
        else:
            numerical_type = 'real'
        i = header.index('Variables:')
        ivar = 0
        for line in header[i+1:]:
            _, name, var_type = line.lstrip().split('\t')
            if ivar == 0:
                self.axis = Axis(name, var_type, self.nPoints, numerical_type)
                self._traces.append(self.axis)
            else:
                trace = Trace(name, var_type, self.nPoints, self.axis, numerical_type)
                self._traces.append(trace)
            ivar += 1       
        self.block_size = (raw_file_size - self.binary_start) // self.nPoints
        self.data_size = self.block_size//self.nVariables
        if self.data_size == 8:
            for point in range(self.nPoints):
                for var in self._traces:
                    value = raw_file.read(8)
                    var.set_pointB8(point, value)
        elif self.data_size == 16:
            for point in range(self.nPoints):
                for var in self._traces:
                    value = raw_file.read(16)
                    var.set_pointB16(point, value)
        else:
            for point in range(self.nPoints):
                        value = raw_file.read(8)  # first variable (ex:time) is always 8 bytes
                        self._traces[0].set_pointB8(point, value)
                        for var in self._traces[1:]:
                            value = raw_file.read(4)
                            var.set_pointB4(point, value)
        raw_file.close()
        self.raw_params["No. Points"] = self.nPoints
        self.raw_params["No. Variables"] = self.nVariables
        self.raw_params["Variables"] = [var.name for var in self._traces]
        if "stepped" in self.raw_params["Flags"]:
            self._load_step_information(raw_filename)

    def get_raw_property(self, property_name=None):
        """Get a property. By default it returns everything"""
        if property_name is None:
            return self.raw_params
        elif property_name in self.raw_params.keys():
            return self.raw_params[property_name]
        else:
            return "Invalid property. Use %s" % str(self.raw_params.keys())

    def get_trace_names(self):
        return [trace.name for trace in self._traces]

    def get_trace(self, trace_ref):
        """Retrieves the trace with the name given. """
        if isinstance(trace_ref, str):
            for trace in self._traces:
                if trace_ref == trace.name:
                    # assert isinstance(trace, DataSet)
                    return trace
            return None
        else:
            return self._traces[trace_ref]

    def _load_step_information(self, filename):
        # Find the extension of the file
        if not filename.endswith(".raw"):
            raise LTSPiceReadException("Invalid Filename. The file should end with '.raw'")
        logfile = filename[:-3] + 'log'
        try:
            log = open(logfile, 'r')
        except:
            raise LTSPiceReadException("Step information needs the '.log' file generated by LTSpice")
        for line in log:
            if line.startswith(".step"):
                step_dict = {}
                for tok in line[6:-1].split(' '):
                    key, value = tok.split('=')
                    try:
                        # Tries to convert to float for backward compatibility
                        value = float(value)
                    except:
                        pass
                        # Leave value as a string to accomodate cases as temperature steps
                        # Temperature steps have the form '.step temp=25°C'
                    step_dict[key] = value
                if self.steps is None:
                    self.steps = [step_dict]
                else:
                    self.steps.append(step_dict)
        log.close()
        if not (self.steps is None):
            # Individual access to the Trace Classes, this information is stored in the Axis
            # which is always in position 0
            self._traces[0]._set_steps(self.steps)
            pass

    def __getitem__(self, item):
        """Helper function to access traces by using the [ ] operator."""
        return self.get_trace(item)

    def get_steps(self, **kwargs):
        if self.steps is None:
            return [0]  # returns an single step
        else:
            if len(kwargs) > 0:
                ret_steps = []  # Initializing an empty array
                i = 0
                for step_dict in self.steps:
                    for key in kwargs:
                        ll = step_dict.get(key, None)
                        if ll is None:
                            break
                        elif kwargs[key] != ll:
                            break
                    else:
                        ret_steps.append(i)  # All the step parameters match
                    i += 1
                return ret_steps
            else:
                return range(len(self.steps))  # Returns all the steps
