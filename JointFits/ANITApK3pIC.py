# This file includes a collection of functions to compute
# quantities related to joint test statistics among combinations
# of three experiments: KM3NeT, ANITA-IV and IceCube

import os
import sys

homedir = os.path.realpath(__file__)[:-len('JointFits/ANITApK3pIC.py')]
sys.path.append(homedir)

datadir = homedir + 'Data/'

import Parameters as pars

import numpy as np
import Experiments.ANITA as AN
import Experiments.KM3NeT as K3
import Experiments.IceCube as IC
import MyUnits as muns
from scipy.optimize import minimize

# ---------------------------------------------#
#                TEST STATISTICS               #
# ---------------------------------------------#

def DiffuseTestStatisticANITA(xs, lt, br, phi):
    """
    Returns the ANITA-IV test statistic.
    """
    AN_chi2 = AN.DiffuseTotalTestStatisticFromPhiInterp(xs,lt,phi,br)
    return AN_chi2

def DiffuseTestStatisticK3(xs, lt, br, phi):
    """
    Returns the KM3NeT test statistic.
    """
    K3_chi2 = K3.DiffuseEventTestStatisticFromPhi(xs,lt,br,phi)
    return K3_chi2

def DiffuseTestStatisticIC(xs, lt, br, phi):
    """
    Returns the IceCube test statistic.
    """
    IC_chi2 = IC.DiffuseEventTestStatisticFromPhi(xs,lt,br,phi) 
    return IC_chi2

def DiffuseTestStatisticAI(xs, lt, br, phi0, return_sum = True):
    """
    Returns the test statistic of ANITA+IceCube.
    If return_sum = False, returns them separately.
    """
    AN_chi2 = DiffuseTestStatisticANITA(xs,lt,br,phi0)
    IC_chi2 = DiffuseTestStatisticIC(xs,lt,br,phi0)
    
    chi2 = np.array([AN_chi2,IC_chi2])
    if return_sum == True:
        return AN_chi2 + IC_chi2
    else:
        return chi2

def DiffuseTestStatisticA3(xs, lt, br, phi0, return_sum = True):
    """
    Returns the test statistic of ANITA+KM3NeT.
    """
    AN_chi2 = DiffuseTestStatisticANITA(xs,lt,br,phi0)
    K3_chi2 = DiffuseTestStatisticK3(xs,lt,br,phi0)
    
    chi2 = np.array([AN_chi2,K3_chi2])
    if return_sum == True:
        return AN_chi2 + K3_chi2
    else:
        return chi2

def DiffuseTestStatisticI3(xs, lt, br, phi0, return_sum = True):
    """
    Returns the test statistic of IceCube+KM3NeT.
    """
    K3_chi2 = DiffuseTestStatisticK3(xs,lt,br,phi0)
    IC_chi2 = DiffuseTestStatisticIC(xs,lt,br,phi0)
    
    if return_sum == True:
        return K3_chi2 + IC_chi2
    else:
        chi2 = np.array([K3_chi2, IC_chi2])
        return chi2

def DiffuseTestStatisticAI3(xs, lt, br, phi0, return_sum = True):
    """
    Returns the test statistic of ANITA-IV+IceCube+KM3NeT.
    """
    AN_chi2 = DiffuseTestStatisticANITA(xs,lt,br,phi0)
    K3_chi2 = DiffuseTestStatisticK3(xs,lt,br,phi0)
    IC_chi2 = DiffuseTestStatisticIC(xs,lt,br,phi0)
    
    if return_sum == True:
        return AN_chi2 + K3_chi2 + IC_chi2
    else:
        chi2 = np.array([AN_chi2,K3_chi2, IC_chi2])
        return chi2

def BestFitDiffuseFluxANITA(xs,lt,br):
    """
    Returns the best-fit flux normalization (phi) for ANITA-IV.
    """
    pred_events_AN = AN.interp_aeff_tot.ev(xs,lt)*AN.LiveTime*(1.-br)
    return AN.number_of_AAEs/pred_events_AN/pars.DiffuseNorm

def BestFitDiffuseFluxKM3NeT(xs,lt,br):
    """
    Returns the best-fit flux normalization (phi) for KM3NeT
    """
    pred_muons_K3  = K3.interp_aeff_mu.ev(xs,lt)*K3.LiveTime*br

    if K3.count_cascades and pars.N_detectable: # KM3NeT looks for cascades and they are detectable
        pred_NT_K3 =    K3.interp_aeff_NT.ev(xs,lt)*K3.LiveTime
    elif K3.count_cascades:  # KM3NeT looks for cascades but the primary vertex is not detectable
        pred_NT_K3 =    K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime
    else:  # KM3NeT does not look for cascades or the primary vertex is not detectable
        pred_NT_K3 = br*K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime
    
    return K3.number_of_muons/(pred_muons_K3+pred_NT_K3)/pars.DiffuseNorm


def BestFitDiffuseFluxA3(xs,lt,br):
    """
    Returns the best-fit flux normalization (phi) for KM3NeT+ANITA.
    """
    pred_events_AN = AN.interp_aeff_tot.ev(xs,lt)*AN.LiveTime*(1.-br)

    pred_muons_K3  = K3.interp_aeff_mu.ev(xs,lt)*K3.LiveTime*br
    if K3.count_cascades and pars.N_detectable: # KM3NeT looks for cascades and they are detectable
        pred_NT_K3 =    K3.interp_aeff_NT.ev(xs,lt)*K3.LiveTime
    elif K3.count_cascades:  # KM3NeT looks for cascades but the primary vertex is not detectable
        pred_NT_K3 =    K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime
    else:  # KM3NeT does not look for cascades or the primary vertex is not detectable
        pred_NT_K3 = br*K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime
    
    return (AN.number_of_AAEs+K3.number_of_muons)/(pred_events_AN+pred_muons_K3+pred_NT_K3)/pars.DiffuseNorm

def BestFitDiffuseFluxAI(xs,lt,br):
    """
    Returns the best-fit flux normalization (phi) for IceCube+ANITA.
    """
    pred_events_AN = AN.interp_aeff_tot.ev(xs,lt)*AN.LiveTime*(1.-br)

    pred_muons_IC = IC.interp_aeff_mu.ev(xs,lt)*IC.LiveTime*br
    pred_NT_IC    = IC.interp_aeff_NT.ev(xs,lt)*IC.LiveTime # all detectable

    return (AN.number_of_AAEs+K3.number_of_muons)/(pred_events_AN+pred_muons_IC+pred_NT_IC)/pars.DiffuseNorm

def BestFitDiffuseFluxI3(xs,lt,br):
    """
    Returns the best-fit flux normalization (phi) for IceCube+KM3NeT.
    """
    pred_muons_K3  = K3.interp_aeff_mu.ev(xs,lt)*K3.LiveTime*br
    if K3.count_cascades and pars.N_detectable: # KM3NeT looks for cascades and they are detectable
        pred_NT_K3 =    K3.interp_aeff_NT.ev(xs,lt)*K3.LiveTime
    elif K3.count_cascades:  # KM3NeT looks for cascades but the primary vertex is not detectable
        pred_NT_K3 =    K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime
    else:  # KM3NeT does not look for cascades or the primary vertex is not detectable
        pred_NT_K3 = br*K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime

    pred_muons_IC  = K3.interp_aeff_mu.ev(xs,lt)*IC.LiveTime*br
    pred_NT_IC     = K3.interp_aeff_NT.ev(xs,lt)*IC.LiveTime
    den = pred_muons_K3+pred_NT_K3+pred_NT_IC+pred_muons_IC

    return K3.number_of_muons/den/pars.DiffuseNorm 


def BestFitDiffuseFluxAI3(xs,lt,br):
    """
    Returns the best-fit flux normalization (phi) for ANITA-IV+IceCube+KM3NeT.
    """
    pred_events_AN = AN.interp_aeff_tot.ev(xs,lt)*AN.LiveTime*(1.-br)

    pred_muons_K3  = K3.interp_aeff_mu.ev(xs,lt)*K3.LiveTime*br
    if K3.count_cascades and pars.N_detectable: # KM3NeT looks for cascades and they are detectable
        pred_NT_K3 =    K3.interp_aeff_NT.ev(xs,lt)*K3.LiveTime
    elif K3.count_cascades:  # KM3NeT looks for cascades but the primary vertex is not detectable
        pred_NT_K3 =    K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime
    else:  # KM3NeT does not look for cascades or the primary vertex is not detectable
        pred_NT_K3 = br*K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime

    pred_muons_IC  = K3.interp_aeff_mu.ev(xs,lt)*IC.LiveTime*br
    pred_NT_IC     = K3.interp_aeff_NT.ev(xs,lt)*IC.LiveTime
    den = pred_events_AN+pred_muons_K3+pred_NT_K3+pred_NT_IC+pred_muons_IC

    return (AN.number_of_AAEs+K3.number_of_muons)/den/pars.DiffuseNorm # this is the differential flux (over 4pi)


def DiffuseTestStatisticBestFitNoPhiA3(xs, lt, br, return_sum = True):
    """
    Returns the best-fit flux normalization (phi) for KM3NeT+ANITA.
    phi is already marginalized.
    """
    phiBF = BestFitDiffuseFluxA3(xs,lt,br)
    return DiffuseTestStatisticA3(xs,lt,br,phiBF,return_sum)

def DiffuseTestStatisticBestFitNoPhiI3(xs, lt, br, return_sum = True):
    """
    Returns the best-fit flux normalization (phi) for KM3NeT+IceCube.
    phi is already marginalized.
    """
    phiBF = BestFitDiffuseFluxI3(xs,lt,br)
    return DiffuseTestStatisticI3(xs,lt,br,phiBF,return_sum)

def DiffuseTestStatisticBestFitNoPhiAI3(xs, lt, br, return_sum = True):
    """
    Returns the best-fit flux normalization (phi) for IceCube+KM3NeT+ANITA.
    phi is already marginalized.
    """
    phiBF = BestFitDiffuseFluxAI3(xs,lt,br)
    return DiffuseTestStatisticAI3(xs,lt,br,phiBF,return_sum)

def BestFitBranchingRatioNoPhiA3(xs,lt):
    """
    Returns the best-fit branching ratio (br) for KM3NeT+ANITA.
    phi is already marginalized.
    Corresponds to equation (X.XX) in 2509.XXXXX
    """
    pred_events_AN = AN.interp_aeff_tot.ev(xs,lt)*AN.LiveTime

    pred_muons_K3  = K3.interp_aeff_mu.ev(xs,lt)*K3.LiveTime

    # if K3.count_cascades and pars.N_detectable: # KM3NeT looks for cascades and they are detectable
    #     pred_NT_K3 =    K3.interp_aeff_NT.ev(xs,lt)*K3.LiveTime
    # elif K3.count_cascades:  # KM3NeT looks for cascades but the primary vertex is not detectable
    #     pred_NT_K3 =    K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime
    # else:  # KM3NeT does not look for cascades or the primary vertex is not detectable
    #     pred_NT_K3 = br*K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime

    # I need to check this!
    if K3.count_cascades:
        pred_NT_K3     = K3.interp_aeff_NT.ev(xs,lt)*K3.LiveTime
        return 1./(1.+AN.number_of_AAEs/K3.number_of_muons*pred_muons_K3/(pred_NT_K3+pred_events_AN))
    else:
        pred_T_K3     = K3.interp_aeff_T.ev(xs,lt)*K3.LiveTime
        return 1./(1.+AN.number_of_AAEs/K3.number_of_muons*(pred_muons_K3+pred_T_K3)/pred_events_AN)

def BestFitBranchingRatioNoPhiAI3(xs,lt):
    """
    Returns the best-fit branching ratio (br) for IceCube+KM3NeT+ANITA.
    phi is already marginalized.
    """
    pred_events_AN = AN.interp_aeff_tot.ev(xs,lt)*AN.LiveTime

    pred_muons_IC  = K3.interp_aeff_mu.ev(xs,lt)*IC.LiveTime
    pred_NT_IC     = K3.interp_aeff_NT.ev(xs,lt)*IC.LiveTime

    pred_muons_K3  = K3.interp_aeff_mu.ev(xs,lt)*K3.LiveTime

    if K3.count_cascades:
        pred_NT_K3     = K3.interp_aeff_NT.ev(xs,lt)*K3.LiveTime
        num = pred_muons_K3+pred_NT_K3+pred_muons_IC+pred_NT_IC
        den = pred_events_AN+pred_NT_K3+pred_NT_IC
    else:
        pred_T_K3 = K3.interp_aeff_NT.ev(xs,lt)*K3.LiveTime
        num = pred_muons_K3+pred_T_K3+pred_muons_IC+pred_NT_IC
        den = pred_events_AN+pred_NT_IC
        
    return 1./(1.+AN.number_of_AAEs/K3.number_of_muons*num/den)


def DiffuseTestStatisticBestFitA3(xs,lt,return_sum = True):
    """
    Returns the best-fit flux normalization (phi) for KM3NeT+ANITA.
    phi and br are already marginalized.
    """
    BrBf  = BestFitBranchingRatioNoPhiA3(xs,lt)
    PhiBf = BestFitDiffuseFluxA3(xs,lt,BrBf)

    return DiffuseTestStatisticA3(xs,lt,BrBf,PhiBf,return_sum)


def DiffuseTestStatisticBestFitI3(xs,lt,return_sum = True):
    """
    Returns the best-fit flux normalization (phi) for IceCube+KM3NeT.
    phi and br are already marginalized.
    """
    BrBf  = 1.0
    PhiBf = BestFitDiffuseFluxI3(xs,lt,BrBf)

    return DiffuseTestStatisticI3(xs,lt,BrBf,PhiBf,return_sum)


def DiffuseTestStatisticBestFitAI3(xs,lt,return_sum = True):
    """
    Returns the best-fit flux normalization (phi) for IceCube+KM3NeT+ANITA.
    phi and br are already marginalized.
    """
    BrBf  = BestFitBranchingRatioNoPhiAI3(xs,lt)
    PhiBf = BestFitDiffuseFluxAI3(xs,lt,BrBf)

    return DiffuseTestStatisticAI3(xs,lt,BrBf,PhiBf,return_sum)