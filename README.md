# numpy_ltspice_filter

A python module for seamless integration of analog filters designed in LTspice into Python3/Numpy
signal processing projects.

run ./filter_demo.py
and ./fast_convolution_filter_demo.py

For a demonstration of its uses.


find an extensive article about this software example at:
https://acidbourbon.wordpress.com/2019/11/26/seamless-integration-of-ltspice-in-python-numpy-signal-processing/

and a follow-up article about fast convolution filtering using LTspice:
https://acidbourbon.wordpress.com/2019/12/04/ltspice-numpy-part-2-fast-convolution-filter/


The utilities in this repository heavily rely on

* __LTSpiceRaw_Reader.py__
A pure python class that serves to read raw files into a python class.

as part of https://github.com/nunobrum/PyLTSpice
developed by Nuno Brum

This repository is included here as a git submodule.
A warm thank you to Nuno.
