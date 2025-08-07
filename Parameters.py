# This file contains parameters of the BSM which can be custom modified.
import numpy as np
import MyUnits as muns


# **********************************************************
# ----------------------------------------------------------
# INFORMATION ON THE BSM
# ----------------------------------------------------------
# **********************************************************

y_NtoT = 0.2 # inelasticity of the N interaction, aka E_T = y_NtoT*E_N
y_Ttomu = 0.5 # inelasticity of the T decay, aka E_mu = y_Ttomu*E_T
y_TtoN = 0.5 # inelasticity of the T decay to the tertiary N, aka E_N' = y_TtoN*E_T

is_T_absorbed = True # is the T absorbed?
N_detectable = True # is the primary vertex detectable?


# **********************************************************
# ----------------------------------------------------------
# INFORMATION ON THE INCOMING FLUX
# ----------------------------------------------------------
# **********************************************************

# What is the reasonable duration of a transient? Not used in the analysis
TransientDuration = 1*muns.day

# With this factor we obtain the flux per unit of solid angle
DiffuseNorm = 1/(4*np.pi)

# IN CASE WE WANT A DELTA FUNCTION FOR THE POWER LAW FLUX, AT WHICH ENERGY?
EnergyDelta = 1*muns.EeV/y_NtoT # this produces an EeV event at ANITA for sure
