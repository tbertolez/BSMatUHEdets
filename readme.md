# BSM at UHE detectors

This is a companion code to 2509.XXXXX (and also to 2305.03746).

A part from basic Python libraries, requires Cython to run (it has been proved to work under Python3.9). First download requires to run

```
python3 setup.py build_ext --inplace 
```

Minor bugs may appear, feel free to open an issue or contact the author, antoni.bertolez@fqa.ub.edu.

The main folder contains:
 - Parameters.py: variables controlling BSM parameters such as the BSM details or the energy of the incoming flux.
 - MyUnits.py: defines the units of the code (cm, s, g, eV).
 - MyFunctions.py, PlottingVariables.py, CyFunctions.pyx: helpful functions.

The main scripts can be found in the following folders:
 - Experiments/: contains the computations for effective areas of ANITA-IV, KM3NeT, IceCube and P-ONE.
 - JointFits/: contains test statistics for individual and joint fits.
 - Calculators/: contains the BSM probabilities, a calculator for effective areas tables, and a calculator for topographic calculations "DistancesFromX.py".
 - Notebooks/: some companion Jupyter notebooks to reproduce the figures of the papers.
 - Data/ and PlotData/: keeps data of the experiments and for plotting.



