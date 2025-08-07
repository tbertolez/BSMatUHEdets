# This file includes a collection of functions to compute
# quantities related to the IceCube experiment. 
# For instance: probabilities, effective areas, 
# number of events or likelihoods. 

import os
import sys

homedir = os.path.realpath(__file__)[:-len('Experiments/IceCube.py')]
sys.path.append(homedir)


datadir = homedir + 'Data/'

import Calculators.Probabilities as probs
import MyUnits as muns
import MyFunctions as mfuns
import CyFunctions as cfuns
import Parameters as pars

import numpy as np
from scipy import integrate

# **********************************************************
# ----------------------------------------------------------
# INFORMATION ON THE ICECUBE DETECTOR
# ----------------------------------------------------------
# **********************************************************

Depth = 2.0*muns.km # the center of the detector
Radius = 0.5642*muns.km
Height = 1.0*muns.km
Volume = np.pi*Radius**2*Height # 1 km^3 exactly

LiveTime = 4605*muns.day # 12.6yrs of data, 2502.01963 
Efficiency = 1.0 # we adjust the official effective area

SectionArea = 1.0*muns.km**2
NumberOfTargets = muns.IceDensity*muns.Avogadro*Volume

MinZenAngle = 0
MaxZenAngle = np.pi

MinExitAngle =  mfuns.get_IC_exit_angle_from_nadir_angle(MinZenAngle,Depth)
MaxExitAngle =  mfuns.get_IC_exit_angle_from_nadir_angle(MaxZenAngle,Depth)

minEnergy = 10*muns.PeV
maxEnergy = pars.y_Ttomu*pars.y_NtoT*pars.EnergyDelta

# First, we compute the nadir angle of the ANITA-IV events as seen from IceCube.
# Check Dropbox/ANITA-IV and neutrinos/Transients/IceCubeCoords.py
AAEs_ZenAngle = np.array([90.0 -15.27, 90.0 + 6.21, 90.0 - 1.96, 90.0 - 3.77])*muns.degs

# ---------------------------------------------#
#          CALCULUS OF PROBABILITIES           #
# ---------------------------------------------#

# In all the following functions, two possible angular quantities are allowed,
# which must be specified as an argument in which_ang:
#   - 'nadir': "angle" is the zenithal angle from IceCube, measured from the nadir
#   - 'exit': "angle" is the zenithal angle as seen from the exit point of the particle
# In the case where "exit" is provided, the hemisphere where the particle came from is needed.
# We assume axial symmetry.

def Layers(angle, which_ang = 'nadir', hemisphere = 'north'):
    """
    Returns the layers that the particle has crossed, in order of first to last.
    Each row contains: [start of the layer, end of layer, width, mass density]
    """
    nad, ext = mfuns.icecube_convert_angle(angle,which_ang,Depth,hemisphere)
    dist_to_exit = mfuns.IC_DistanceToExitPoint(nad,Depth)
    lays = cfuns.LayersThroughChord(ext,dist_to_exit)
    if lays.shape[0] == 1:
        lays = np.vstack((lays,lays))
    return lays

def length(angle, which_ang = 'nadir', hemisphere = 'north'):
    """ 
    Returns the total distance travelled by the particle
    """
    lays = Layers(angle,which_ang,hemisphere)
    return lays[-1,1]
 
def ProbMuon_dmin(angle,einiN,xs,lt, which_ang = 'nadir', hemisphere = 'north'):
    """
    Returns the probability that a muon arrives to the detector
    with energy larger than 10PeV, in the direction of angle.
    Check Parameters.py to activate/deactivate T absorption through pars.is_T_absorbed
    """
    lay = Layers(angle,which_ang,hemisphere)
    return probs.ProbMuon_dmin(lay,einiN*pars.y_NtoT*pars.y_Ttomu,xs,lt,pars.is_T_absorbed)

# def ProbMuon(angle,efin,einiN,xs,lt, which_ang = 'nadir', hemisphere = 'north'):
#     Returns the probability that a muon arrives to the detector with energy efin.
#     lay = Layers(angle,which_ang,hemisphere)
#     return probs.ProbMuon(lay,efin,einiN,xs,lt,pars.is_T_absorbed)

def ProbN(angle,efin, einiN, xs, lt, which_ang = 'nadir', 
                                     hemisphere = 'north'):
    """ 
    Probability a particle N exits the Earth with energy efin.
    Implements equation (A.19) from 2305.03746.
    """
    lay = Layers(angle,which_ang,hemisphere)
    return probs.ProbN(lay,efin,einiN,xs,lt,pars.is_T_absorbed)

def ProbT(angle,efin, einiN, xs,lt, which_ang = 'nadir', 
                                    hemisphere = 'north'):
    """ 
    Probability a particle T exits the Earth with energy efin.
    """
    lay = Layers(angle,which_ang,hemisphere)
    return probs.ProbT(lay,efin,einiN,xs,lt,pars.is_T_absorbed)

def DecayProbabilityT(lt):
    """
    Probability a T particle decays inside the detector (once arrived).
    """
    return probs.ProbDecayIn(lt,2*Radius)


# ---------------------------------------------#
#               EFFECTIVE AREAS                #
# ---------------------------------------------#

def GeometricArea():
    """
    Returns the geometric area of the detector
    """
    return SectionArea

# In the following functions, pars.EnergyDelta is the energy of the initial flux of N particles.
# Then, in principle the events can have different final energies efin.

def EffectiveAreaMuon(angle, xs, lt):
    """
    Returns the effective area to through-going muons,
    considering all muons with final energy above 10PeV.
    Implements equation (XX.X) from 2509.XXXXX
    """
    A_geom = GeometricArea()
    return Efficiency*A_geom*ProbMuon_dmin(angle,pars.EnergyDelta,xs, lt)

def EffectiveAreaT(angle,efin, xs,lt):
    """
    Returns the effective area to T decays (and maybe T scattering too)    .
    Implements equation (XX.X) from 2509.XXXXX
    """
    probT = ProbT(angle,efin,pars.EnergyDelta,xs,lt)
    area = GeometricArea()*DecayProbabilityT(lt)
    if pars.is_T_absorbed:
        area += NumberOfTargets*xs
    return probT*area*Efficiency

def EffectiveAreaN(angle,efin, xs,lt):
    """
    Returns the effective area to N scattering.    
    Implements equation (XX.X) from 2509.XXXXX
    """
    probN = ProbN(angle, efin, pars.EnergyDelta, xs, lt)
    return probN*NumberOfTargets*xs*Efficiency

def EffectiveArea(angle,efin, xs,lt):
    """
    Returns the total effective area in the direction of angle. 
    """
    return (EffectiveAreaMuon(angle,     xs,lt)+
            EffectiveAreaT(   angle,efin,xs,lt) + 
            EffectiveAreaN(   angle,efin,xs,lt))

def TotalEffectiveAreaMuon(xs,lt,eps=1e-2):
    """
    Returns the total effective area to muons, all-sky integrated.
    """
    def fsmooth(elev):
        return np.sin(elev)*EffectiveAreaMuon(elev,xs,lt) 
    intsmoo = 2*np.pi*integrate.quad(fsmooth, MinZenAngle,MaxZenAngle, epsrel = eps)[0]
    return intsmoo


def TotalEffectiveAreaN(xs,lt, eps = 1e-2):
    """
    Returns the total effective area to N, all-sky integrated.
    If our initial flux is a delta function in energies, we do not need an integration in energy (since initial and final energies are related by a delta)
    """
    def fsmooth(elev):
        return np.sin(elev)*(EffectiveAreaN(elev,pars.EnergyDelta,xs,lt)+
                             EffectiveAreaN(elev,pars.EnergyDelta*pars.y_NtoT*pars.y_TtoN,xs,lt)) 
    intsmoo = 2*np.pi*integrate.quad(fsmooth,MinZenAngle,MaxZenAngle,epsrel = eps)[0]
    return intsmoo

def TotalEffectiveAreaT(xs,lt, eps = 1e-2):
    """
    Returns the total effective area to T, all-sky integrated.
    If our initial flux is a delta function in energies, we do not need an integration in energy (since initial and final energies are related by a delta)
    """
    def fsmooth(elev):
        return np.sin(elev)*EffectiveAreaT(elev,pars.EnergyDelta*pars.y_NtoT,xs,lt)
    intsmoo = 2*np.pi*integrate.quad(fsmooth,MinZenAngle,MaxZenAngle,epsrel = eps)[0]
    return intsmoo

def TotalEffectiveArea(xs,lt,br):
    """
    Returns the total effective area to all possible signals.
    Muons are only produced if T decays into a muon (given by br),
    while N/T scattering and T decay always produce a signal.
    """
    return br*TotalEffectiveAreaMuon(xs,lt)+TotalEffectiveAreaN(xs,lt)+TotalEffectiveAreaT(xs,lt)


# -----------------------------------------
# Muon energy-dependent probability
# ----------------------------------------

# In this data release we have neglected this part of the code
# because it requires a heavy montecarlo file, which we have to decided
# to leave aside to keep the release light. 
# This code is not necessary for the analysis performed in our papers,
# only for some complementary figure in the appendix.
# Please contact the developers for further info on this code.

# muon_points = np.loadtxt(homedir+"Data/all_muon_data_530PeV.dat")
# muon_points[:,0] *= muns.PeV
# muon_points[:,1] *= muns.km
# xarr = np.unique(muon_points[:,1])

# ebins = np.geomspace(100*muns.TeV,530*muns.PeV,9)
# elow, eupp = ebins[:-1], ebins[1:]

# def ProbMuonTail(angle,eebin,xs,lt, which_ang = 'nadir', 
#                                     hemisphere = 'north'):
#     """
#     This computes the probability for a muon to arrive with energy within the bin eebin
#     """
#     lay = Layers(angle,which_ang,hemisphere) 
#     probhist = 0
#     for j, dmu in enumerate(xarr):
#         pt = probs.ProbMuonAtd(lay,dmu,xs,lt,pars.is_T_absorbed)
#         wj = 1*muns.km*pt
#         subarr = muon_points[np.where(muon_points[:,1]==dmu)]
#         Nj = subarr.shape[0]+1
#         pj = np.histogram(subarr[:,0],eebin)[0]/Nj
#         probhist += wj*pj[0]
    
#     return probhist

# def efficiency_ic(ec):
#     """
#     Returns the energy dependence of IceCube's efficiency.
#     Taken from 1502.02649
#     """
#     c, d, q = 0.5, 1.1, 4.6
#     fsat = 0.453844
#     e_th = 100*muns.TeV
#     x = np.log10(ec/e_th)
#     return (c*x**q/(1.+d*x**q))/fsat

# def EffectiveAreaMuonTail(elev,eebin,xs,lt):
#     return ProbMuonTail(elev,eebin,xs,lt)*Efficiency*efficiency_ic(np.sqrt(eebin[0]*eebin[1]))*GeometricArea()

# def TotalEffectiveAreaMuonTail(eebin,xs,lt,eps =1e-2):
#     def fsmooth(elev):
#         return np.sin(elev)*EffectiveAreaMuonTail(elev,eebin,xs,lt) # could have chosen whichever azimuth
#     intsmoo = 2*np.pi*integrate.quad(fsmooth, MinZenAngle,MaxZenAngle, epsrel = eps)[0]
#     return intsmoo

######################################
# NUMBER OF EVENTS
######################################

def DiffuseEvents(xs,lt,br, 
                        phi = 1/muns.km**2/muns.day, 
                        DT = LiveTime,
                        return_sum = True):
    """
    Returns the total number of events assuming a diffuse flux.
    If return_sum = False, events are returned as
    [through-going muons, cascades, starting tracks]
    """
    fnorm = pars.DiffuseNorm # This is because Phi is integrated for all solid angle, so avoid double-counting
    if return_sum:
        int_area = TotalEffectiveArea(xs,lt,br)
        return DT*phi*fnorm*int_area
    else:
        area_mu = TotalEffectiveAreaMuon(xs,lt)
        area_N  = TotalEffectiveAreaN(xs,lt)
        area_T  = TotalEffectiveAreaT(xs,lt)
        return np.array([DT*phi*fnorm*area_mu*br, # through-going muons
                         DT*phi*fnorm*(area_N+(1-br)*area_T), #cascades
                         DT*phi*fnorm*area_T*br]) # starting tracks

# ---------------------------------------------#
#               INTERPOLATING                  #
# ---------------------------------------------#

# For speeding up computations, if one has already computed total and averaged
# effectives areas (see Calculators/EffectiveAreas.py), interpolating them from
# a data file speeds up the computation of tests statistics.

areatot_fname = datadir+'IceCube/EffectiveAreas_TAbsorption.dat'
interp_aeff_mu   = mfuns.create_interpolator_from_datafile(areatot_fname, 2) # area to muons
interp_aeff_NT   = mfuns.create_interpolator_from_datafile(areatot_fname, (3,4)) # area to N+T
interp_aeff_T    = mfuns.create_interpolator_from_datafile(areatot_fname, 4) # area to T only

def reset_interpolators():
    global interp_aeff_mu, interp_aeff_NT, interp_aeff_T
    # This will work only if areatot_fname is not rewritten!
    interp_aeff_mu   = mfuns.create_interpolator_from_datafile(areatot_fname, 2) 
    interp_aeff_NT   = mfuns.create_interpolator_from_datafile(areatot_fname, (3,4))
    interp_aeff_T    = mfuns.create_interpolator_from_datafile(areatot_fname, 4)
    return

def DiffuseEventTestStatisticFromPhi(xs,lt,br,phi):
    """
    Computes the test statistic (TS) for IceCube given xs, lt, br, phi;
    implements eq. (XX.X) from 2509.XXXXX
    """
    fnorm = pars.DiffuseNorm # factor 1/(4pi)

    if pars.N_detectable:
        int_aeff = br*interp_aeff_mu.ev(xs,lt) + interp_aeff_NT.ev(xs,lt)
    else:
        int_aeff = br*interp_aeff_mu.ev(xs,lt) + interp_aeff_T.ev(xs,lt)

    chi2 = 2*fnorm*phi*LiveTime*int_aeff
    return chi2