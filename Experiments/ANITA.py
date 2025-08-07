# This file includes a collection of functions to compute
# quantities related to the ANITA(-IV) experiment. 
# For instance: probabilities, effective areas, 
# number of events or likelihoods. 
# Further information on our ANITA analysis in 2305.03746

import os
import sys

homedir = os.path.realpath(__file__)[:-len('Experiments/ANITA.py')]
sys.path.append(homedir)

datadir = homedir + 'Data/'

import MyUnits as muns
import MyFunctions as mfuns
import CyFunctions as cfuns
import Calculators.Probabilities as probs
import Parameters as pars

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.interpolate import interp1d


# ---------------------------------------------#
#               ANITA PARAMETERS               #
# ---------------------------------------------#

MeanHeight = 38.7325*muns.km
LiveTime   = 24.5*muns.day

MinElevAngle = -4.7*muns.degs
MaxElevAngle = -40.*muns.degs

ThetaCone  = 1.*muns.degs

# ANITA-IV ANOMALOUS EVENTS
# ---------------------------------------------

number_of_AAEs = 4
# This comes from https://arxiv.org/pdf/2008.05690 and refers to the particle which decayed
AAEs_Energy = np.array([1.5,0.9,0.8,3.9])*muns.EeV
AAEs_EnergyError = np.array([0.7,0.5,0.3,2.5])*muns.EeV

# AAEs AS SEEN FROM ANITA
# ------------------------------------------

# Angular location of the AAEs
# The last two ones are from ANITA-I and III, in principle they should not be read as long as number_of_AAEs = 4
AAEs_ElevAngleH     = np.array([-5.92, -6.06, -5.92, -5.93, -5.93, -5.93])*muns.degs # Horizon angle as seen from ANITA
AAEs_ElevAngleSubH  = np.array([-0.25, -0.65, -0.81, -0.19, -21.47, -25.07])*muns.degs # Event angle with respect to the horizon, as seen from ANITA
AAEs_ElevAngle      = AAEs_ElevAngleH + AAEs_ElevAngleSubH # Event angle as seen from ANITA
AAEs_ElevAngleError = np.array([ 0.21,  0.20,  0.20,  0.10, 0.3, 0.2])*muns.degs # Event angle error as seen from ANITA

# Height of the ANITA detector at the time of detection
AAEs_Height = np.array([38.86,38.97,38.52,38.58, 38.6, 38.6])*muns.km

AAEs_MeanElevAngle      = -6.43*muns.degs
AAEs_MeanElevAngleError = 0.18*muns.degs

# AAEs AS SEEN FROM THE EXIT POINT TOWARDS ANITA
# -------------------------------------------------------

AAEs_ExitAngle = np.array([mfuns.get_anita_exit_angle_from_elv_angle(AAEs_ElevAngle[i],
                                                                     AAEs_Height[i])
                           for i in range(number_of_AAEs)])
# AAEs_ExitAngle = [88.35239907 86.90242633 86.79477415 88.46624585] (degrees)
AAEs_MeanExitAngle = np.mean(AAEs_ExitAngle)


AAEs_ExitAngleError = np.array([mfuns.get_anita_exit_angle_error(AAEs_ElevAngle[i],
                                                                 AAEs_ElevAngleError[i],
                                                                 AAEs_Height[i])
                           for i in range(number_of_AAEs)])
# AAEs_ExitAngleError = [0.78925259 0.43480979 0.42144104 0.40044971] (degrees)
AAEs_MeanExitAngleError = np.mean(AAEs_ExitAngleError)

# The maximum angle of vision at the moment of AAE detections
AAEs_MinExitAngle = np.array([90*muns.degs]*number_of_AAEs) # the "minimum" exit angle will always be 90º.
AAEs_MaxExitAngle = np.array([mfuns.get_anita_exit_angle_from_elv_angle(MaxElevAngle,h) for h in AAEs_Height])
# AAEs_MaxExitAngle = [50.37087792 50.37193186 50.36762043 50.36819526] (degrees)

AAEs_MeanMinExitAngle = np.mean(AAEs_MinExitAngle)
AAEs_MeanMaxExitAngle = np.mean(AAEs_MaxExitAngle)

# ---------------------------------------------#
#                PROBABILITIES                 #
# ---------------------------------------------#

# In all the following functions, 
#   - xs (float): cross-section of N interaction (and of T absorption maybe)
#   - lt (float): lifetime in s in the labframe, at 1EeV

def Layers(th):
    """
    Returns the layers that the particle has crossed, in order of first to last.
    Each row contains: [start of the layer, end of layer, width, mass density]
    """
    lays = cfuns.LayersThroughChord(th)
    if lays.shape[0] == 1:
        # CyFunctions.py requires that we duplicate the layer if it's only one.
        lays = np.vstack((lays,lays))
    return lays

def ProbN(th,xs,lt):
    """ 
    Probability a particle N exits the Earth. 
    Check Parameters.py to activate/deactivate T absorption through pars.is_T_absorbed
    """
    lay = Layers(th)
    return probs.ProbN(lay,xs,lt,pars.is_T_absorbed)

def ProbT(th,xs,lt):
    """
    Probability a particle T exits the Earth.
    """
    lay = Layers(th)
    return probs.ProbT(lay,xs,lt,pars.is_T_absorbed)

def DecayBeforeProbabilityT(lt, length):
    """
    Probability of a T particle with lifetime lt decays before a distance "length".
    """
    return 1 - np.exp(-length/(muns.c*lt))

def TriggerProbability(angle): # angle is the elevation angle
    """
    Probability a cascade from an angle triggers ANITA's antennae.
    Interpolated from Fig. 10 in 2112.07069
    """
    prob  = mfuns.get_data(datadir+"ANITA/Ptrig.dat")  # x in degrees
    return np.interp(angle, prob[:,0]*muns.degs, prob[:,1], left = 0.0, right = 0.0)


# ---------------------------------------------#
#               EFFECTIVE AREAS                #
# ---------------------------------------------#

# In all the following functions, two possible angular quantities are allowed,
# which must be specified as an argument in which_ang:
#   - 'elev': "angle" is the elevation angle
#   - 'exit': "angle" is the zenithal angle as seen from the exit point of the particle
# The height of the detector at a specific time can also be specified.


def GeometricArea(angle,height = MeanHeight, which_ang = 'elev'):
    """
    Returns the geometric area as in Fig. 10 in 2112.07069
    """
    Ag = mfuns.get_data(datadir+"ANITA/Ag.dat")
    
    if which_ang == 'elev':
        return np.interp(angle, Ag[:,0]*muns.degs, Ag[:,1]*muns.km**2, left = 0.0, right = 0.0)
    
    elif which_ang == 'exit':
        elv, th = mfuns.anita_convert_angle(angle,which_ang,height)
        return np.interp(elv,   Ag[:,0]*muns.degs, Ag[:,1]*muns.km**2, left = 0.0, right = 0.0)

def GeometricArea_ExitProbT(angle,xs,lt,height = MeanHeight, 
                                        which_ang = 'elev'):
    """
    Computes the geometric area of ANITA times the probability that a T particle exits the Earth.
    """
    try: 
        elv, th = mfuns.anita_convert_angle(angle,which_ang,height)
    except:
        # If the angle is over-horizon, mfuns.anita_convert_angle will raise an error
        # The area over-horizon is zero, we return zero geometric area.
        return 0.

    area = GeometricArea(elv)
    return ProbT(th,xs,lt)*area


def EffectiveAreaT(angle,xs,lt, height = MeanHeight, 
                                which_ang = 'elev'):
    """
    Computes the effective area of ANITA for the contribution of T particles.
    Implements Adec_T and Aint_T from equations (A.7) and (A.11) from 2305.03746
    """

    # We compute the contribution of T particles from decay in the instrumented volume.
    ageo_pexit = GeometricArea_ExitProbT(angle,xs,lt,height,
                                                     which_ang)
    if ageo_pexit == 0:
        # If the geometric area is already zero, there is no point in computing anything else
        return 0.

    elv, th = mfuns.anita_convert_angle(angle,which_ang,height)

    length = mfuns.ANITA_DistanceToExitPoint(elv,height)
    pdecay = DecayBeforeProbabilityT(lt,length)

    # If T can interact with matter, it can also produce signals via xs.
    T_int_contribution = 0.
    if pars.is_T_absorbed:
        n_tgt = NumberOfTargets(elv,height)
        probT = ProbT(th, xs, lt)
        T_int_contribution = n_tgt*xs*probT

    # ANITA efficiency from the trigger probability
    ptrig = TriggerProbability(elv)  

    # Return the sum of the two contributions
    return (ageo_pexit*pdecay+T_int_contribution)*ptrig


def Volume(elev, height):
	th = mfuns.get_anita_exit_angle_from_elv_angle(elev,height)
	dist_to_exit = mfuns.ANITA_DistanceToExitPointSafe(th,elev)
	return 1/3.*np.pi*dist_to_exit**3*np.tan(ThetaCone*muns.degs)

def NumberOfTargets(elev,height):
	"""
	This the number of targets inside the cone from the exit point towards ANITA (as seen in fig. 8 from 2112.07069).
	Probably needs a bit of refining and dependence on the angle,
	but its contribution should be sub-dominant.
	"""
	return muns.AirDensity*muns.Avogadro*Volume(elev,height)


def EffectiveAreaN(angle,xs,lt,height = MeanHeight, 
                               which_ang = 'elev'):
    """
    Computes the effective area of ANITA for N BSM particles.
    Implements A_N from equation (A.11) from 2305.03746.
    """
  
    try: 
        elv, th = mfuns.anita_convert_angle(angle,which_ang,height)
    except:
        # If the angle is over-horizon, mfuns.anita_convert_angle will raise an error
        # The area over-horizon is zero, we return zero contribution from N particles.
        return 0.

    ptrig = TriggerProbability(elv)
    probN = ProbN(th, xs, lt)
    aeffN = probN*NumberOfTargets(elv,height)*xs*ptrig

    return aeffN


def EffectiveArea(angle,xs,lt,height = MeanHeight, 
                              which_ang = 'elev'):
    """
    Computes the effective area of ANITA for T and N BSM particles.
    Implements the sum of all terms in equations (A.7) and (A.11) from 2305.03746.
    """
    aeffT = EffectiveAreaT(angle,xs,lt,height,
                                       which_ang)
    
    aeffN = EffectiveAreaN(angle,xs,lt,height,
                                       which_ang)

    return aeffN + aeffT

def AverageEffectiveArea(xs,lt,threc  = AAEs_MeanExitAngle,
                               dth    = AAEs_MeanExitAngleError,
                               height = MeanHeight,
                               thmin  = AAEs_MeanMinExitAngle,
                               thmax  = AAEs_MeanMaxExitAngle,
                               eps = 1e-3):
    """
    Averages the effective area with a gaussian around threc and uncertainty dth.
    Implements equation (B.9) from 2305.03746.
    
    Extra input:
        threc, dth (float): reconstructed angle of the event by ANITA and its angular uncertainty.
        thmin, thmax (float): the angular limits of the integral. By default, all the ANITA aperture.
    """

    def exp_weight(x,mu,sig):
        return np.exp(-(x-mu)**2/2/sig**2)/np.sqrt(2*np.pi)/dth

    def f(th):
        return np.sin(th)*exp_weight(th,threc,dth)*EffectiveArea(th,xs,lt,height,
                                                                       'exit')

    int, err = integrate.quad(f,thmax,thmin,epsrel = eps)
    return int


def TotalEffectiveArea(xs, lt, height = MeanHeight,
                               amin   = AAEs_MeanMaxExitAngle,
                               amax   = AAEs_MeanMinExitAngle,
                               eps = 1e-3):
    """
    Integrates the effective area in the full solid angle between exit angles amin and amax. 
    Implements the definition A^tot from 2305.03746.
    """
    
    def f(nad):
        return np.sin(nad)*EffectiveArea(nad, xs, lt, height, 
                                                      'exit')

    int, err = integrate.quad(f,amin,amax,epsrel = eps)
    return 2*np.pi*int


# ---------------------------------------------#
#               NUMBER OF EVENTS               #
# ---------------------------------------------#

# In the following functions,
#   - phi is the flux normalization
#   - DT is the time during which the flux can be observed by ANITA (by default, all the time of flight)

def TransientEvents(xs, lt, angle, phi = 1/muns.cm**2/muns.s,
                                   height = MeanHeight,
                                   DT = LiveTime,
                                   which_ang = 'elev'):
    """
    Computes the total number of events in ANITA
    for a transient event from a certain angle.
    """
    return DT*phi*EffectiveArea(angle,xs,lt, height = height,
                                             which_ang    = which_ang)


def DiffuseEvents(xs, lt, phi = 1/muns.km**2/muns.day,
                          DT = LiveTime):
    """
    Computes the total number of events in ANITA
    for an isotropic and constant ("diffuse") flux.

    Implements the sum of equations (A.10), (A.13), (A.14) from 2305.03746,
    integrated in solid angle for f_Omega = 1/4pi.
    """

    fnorm = pars.DiffuseNorm # 1/4pi
    return DT*phi*fnorm*TotalEffectiveArea(xs,lt)


# ---------------------------------------------#
#               INTERPOLATING                  #
# ---------------------------------------------#

# For speeding up computations, if one has already computed total and averaged
# effectives areas (see Calculators/EffectiveAreas.py), interpolating them from
# a data file speeds up the computation of tests statistics.

namefile = datadir+'ANITA/EffectiveAreas.dat'
all_aeff_data = mfuns.get_data(datadir+'ANITA/EffectiveAreas.dat')
# First column is the total effective area, 
# following columns are the effective area averaged at each of the AAEs.

interp_aeff_tot  = mfuns.create_interpolator_from_datafile(namefile,2) # total Aeff
interp_avg1_aeff = mfuns.create_interpolator_from_datafile(namefile,3) # AAE1
interp_avg2_aeff = mfuns.create_interpolator_from_datafile(namefile,4) # AAE2
interp_avg3_aeff = mfuns.create_interpolator_from_datafile(namefile,5) # AAE3
interp_avg4_aeff = mfuns.create_interpolator_from_datafile(namefile,6) # AAE4
interps_avgs = [interp_avg1_aeff,interp_avg2_aeff,interp_avg3_aeff,interp_avg4_aeff]

# ---------------------------------------------#
#                TEST STATISTICS               #
# ---------------------------------------------#

# branching ratio (br) per default is 0, so that all T decays go into cascades

def DiffuseEventTestStatisticFromPhi(i, xs,lt,phi, br = 0.):
    """
    Computes the test statistic (TS) for ANITA given xs, lt and phi;
    assuming an isotropic and constant flux, for a single event.
    Implements equation (B.8) for N = 1 from 2305.03746.
    
    Extra input:
        i (int 0-3): the event we are computing the TS of.
    """
    dth = AAEs_ExitAngleError[i]   # Angular uncertainty of the event
    h   = AAEs_Height[i]           # Height of the antenna at the time of detection
    thr = AAEs_ExitAngle[i]        # Reconstructed angle from ANITA
    Dt  = LiveTime                 # Total ANITA livetime
    fnorm = pars.DiffuseNorm       # Factor 1/4pi

    # Compute the total effective area, integrated per all solid angle
    int_aeff = (1.-br)*TotalEffectiveArea(xs,lt,h)
    
    # Compute Abar, equation (B.9). It is not integrated in solid angle.
    avg_aeff = (1.-br)*AverageEffectiveArea(xs,lt,thr,dth,h)

    # We compute the different terms of the TS
    chi2  = 0. # This term is not necessary, the likelihood is indifferent to the addition of constants.
    chi2 -= 2*np.log(fnorm*phi*Dt*2*np.pi*avg_aeff) # fnorm = 1/4pi
    chi2 += 2*fnorm*phi*Dt*int_aeff
    return chi2


def DiffuseTotalTestStatisticFromPhi(xs,lt,phi, br = 0.):
    """
    Computes the test statistic (TS) for ANITA given xs, lt and phi;
    assuming an isotropic and constant flux, for all the events.
    Implements equation (B.8) from 2305.03746.
    """
    Dt  = LiveTime  # Total ANITA livetime
    fnorm = pars.DiffuseNorm   # Factor 1/4pi

    chi2 = 0.0
    # Loop of events
    for i in range(number_of_AAEs):
        dth = AAEs_ExitAngleError[i]   # Angular uncertainty of the event
        h   = AAEs_Height[i]           # Height of the antenna at the time of detection
        thr = AAEs_ExitAngle[i]        # Reconstructed angle from ANITA

        # Compute the total effective area, integrated per all solid angle
        int_aeff = (1-br)*TotalEffectiveArea(xs,lt,h)
        
        # Compute \bar{A}, equation (B.9). It is not integrated in solid angle.
        avg_aeff = (1-br)*AverageEffectiveArea(xs,lt,thr,dth,h)
        
        # We compute the different terms of the TS
        chi2 -= 2*np.log(fnorm*phi*Dt*2*np.pi*avg_aeff)
        chi2 += 2*fnorm*phi*Dt*int_aeff/number_of_AAEs
        # In principle, only one contribution from this second term (int_aeff) for all events.
        # But since each AAE has a different height, we do an average of the total expected number of events
        # and therefore divide by pars.number_of_AAEs.

    return chi2

def DiffuseTotalTestStatisticFromPhiInterp(xs,lt,phi, br = 0.):
    """
    Same as before, but interpolating from the data files.
    """
    Dt  = LiveTime  # Total ANITA livetime
    fnorm = pars.DiffuseNorm   # Factor 1/4pi

    chi2 = 0.0
    # Loop of events

    int_aeff = (1.-br)*interp_aeff_tot.ev(xs,lt)
    chi2  = 2*fnorm*phi*Dt*int_aeff
    for i in range(number_of_AAEs):
        avg_aeff = interps_avgs[i].ev(xs,lt)
        chi2 -= 2*np.log(fnorm*phi*Dt*2*np.pi*(1.-br)*avg_aeff)

    return chi2

def BestFitDiffuseFlux(xs,lt,br = 0.):
    """
    Computes the best-fit flux normalization for ANITA;
    assuming an isotropic and constant flux, for all the events.
    Implements equation (B.10) from 2305.03746.
    """
    Dt = LiveTime          # Total ANITA time of observation
    nevents = number_of_AAEs     # Total number of events

    return nevents/pars.DiffuseNorm/Dt/(1.-br)/TotalEffectiveArea(xs,lt)


def DiffuseEventTestStatistic(i,xs,lt):
    """
    Computes the test statistic (TS) for ANITA given xs, lt;
    phi is already marginalized, for a single event.
    Implements equation (B.11) from 2305.03746.
    """
    dth = AAEs_ExitAngleError[i]   # Angular uncertainty of the event
    h   = AAEs_Height[i]           # Height of the antenna at the time of detection
    thr = AAEs_ExitAngle[i]        # Reconstructed angle from ANITA

    # Compute the total effective area, integrated per all solid angle
    int_aeff = TotalEffectiveArea(xs,lt,h)
    
    # Compute Abar, equation (B.9). It is not integrated in solid angle.
    avg_aeff = AverageEffectiveArea(xs,lt,thr,dth,h)

    # remember int_aeff is integrated per all solid angle
    chi2 = -2*np.log(2*np.pi*avg_aeff/int_aeff)
    return chi2


def DiffuseTotalTestStatistic(xs, lt, br = 0.,
                                      return_sum = True):
    """
    Computes the test statistic (TS) for ANITA given xs, lt;
    phi is already marginalized, for all events.
    Implements the sum of equation (B.11) for all events, from 2305.03746.
    
    Extra input:
        return_sum (bool): whether to return the total TS, or an array with the separate contributions.
    """

    # We basically collect the contributions to the TS from each event
    chi2 = []
    for i in range(number_of_AAEs):
        chi2.append(DiffuseEventTestStatistic(i, xs, lt, br))
    
    chi2 = np.array(chi2)
    if return_sum == True:
        # We return the total TS
        return np.sum(chi2)
    else:
        # We return each of the contributions separately in an array
        return chi2
    

def DiffuseTotalTestStatisticInterp(xs,lt, br = 0.):
    """
    Same as before, but interpolated.
    """
    Dt  = LiveTime  # Total ANITA livetime
    fnorm = pars.DiffuseNorm   # Factor 1/4pi

    chi2 = 0.0
    # Loop of events

    int_aeff = (1.-br)*interp_aeff_tot.ev(xs,lt)
    phi = 2*4/int_aeff
    chi2  = 0. 
    for i in range(number_of_AAEs):
        avg_aeff = interps_avgs[i].ev(xs,lt)
        chi2 -= 2*np.log(fnorm*phi*Dt*2*np.pi*(1.-br)*avg_aeff)

    return chi2
