# This script is useful to write tables of total effective areas.
# It integrates the effective area for the full solid angle,
# given BSM parameters, for the different experiments

import os
import time
import sys
sys.path.append("..")
import io
import pstats
homedir = os.path.realpath(__file__)[:-len('Calculators/EffectiveAreas.py')]
sys.path.append(homedir)

import numpy as np
import Experiments.ANITA as ANITA
import Experiments.IceCube as IC
import Experiments.KM3NeT as K3
# import Experiments.KM3NeT_sphere as K3
import Parameters as pars

# In all the following functions, xxs and llt are numpy arrays with all the
# cross sections and lifetimes at which to compute the effective area.

# -----------------
# ANITA FUNCTIONS
# -----------------
def write_ANITA_total_effective_area(filename,xxs,llt):
    """
    Writes a file "filename" with a table of the total ANITA effective area.
    """    
    
    file = open(filename,'w+')
    for xs in xxs:
        for lt in llt:
            aeff = ANITA.TotalEffectiveArea(xs,lt)
            file.write('{0:1.5e},{1:1.5e},{2:1.5e}\n'.format(xs,lt,aeff))
    return 

def write_ANITA_all_effective_areas(filename,xxs,llt):
    """
    Writes a file "filename" with a table of the averaged ANITA effective area for all events.
    """    
    file = open(filename,'w+')
    avg_aeffs = np.zeros(ANITA.number_of_AAEs)
    for xs in xxs:
        for lt in llt:
            aeff = ANITA.TotalEffectiveArea(xs,lt)
            for i in range(ANITA.number_of_AAEs):
                thr   = ANITA.AAEs_ExitAngle[i]
                dth   = ANITA.AAEs_ExitAngleError[i]
                h     = ANITA.AAEs_Height[i]
                thmin = np.amin([ANITA.AAEs_MinExitAngle[i], thr+4*dth]) # This definition is so counter-intuitive
                thmax = np.amax([ANITA.AAEs_MaxExitAngle[i], thr-4*dth])
                avg_aeffs[i] = ANITA.AverageEffectiveArea(xs,lt, thr, dth, h, thmin, thmax)
            
            file.write('{0:1.5e},{1:1.5e},{2:1.5e},{3:1.5e},{4:1.5e},{5:1.5e},{6:1.5e}\n'.format(xs,lt,aeff,avg_aeffs[0],avg_aeffs[1],avg_aeffs[2],avg_aeffs[3]))
            print('{0:1.5e},{1:1.5e},{2:1.5e},{3:1.5e},{4:1.5e},{5:1.5e},{6:1.5e}\n'.format(xs,lt,aeff,avg_aeffs[0],avg_aeffs[1],avg_aeffs[2],avg_aeffs[3]))
    return 


# -----------------
# ICECUBE FUNCTIONS
# -----------------
def write_IC_total_effective_area(filename,xxs,llt):
    """
    Writes a file "filename" with a table of the total IceCube effective area.
    """        
    file = open(filename,'w+')
    for xs in xxs:
        for lt in llt:
            aeff = IC.TotalEffectiveArea(xs,lt)
            file.write('{0:1.5e},{1:1.5e},{2:1.5e}\n'.format(xs,lt,aeff))
    return 

def write_IC_total_effective_areas(filename,xxs,llt, eps = 1e-2):
    """
    Writes a file "filename" with a table of the total IceCube effective area,
    separated among the different contributions (muons, N and T).
    """   
    file = open(filename,'w+')     
    for xs in xxs:
        for lt in llt:
            aeffmu = IC.TotalEffectiveAreaMuon(xs,lt,eps=eps)
            aeffN  = IC.TotalEffectiveAreaN(xs,lt,eps=eps)
            aeffT  = IC.TotalEffectiveAreaT(xs,lt,eps=eps)
            print(IC.maxEnergy)
            file.write('{0:1.5e},{1:1.5e},{2:1.5e},{3:1.5e},{4:1.5e}\n'.format(xs,lt,aeffmu,aeffN,aeffT))
            print('{0:1.5e},{1:1.5e},{2:1.5e},{3:1.5e},{4:1.5e}\n'.format(xs,lt,aeffmu,aeffN,aeffT))
    file.close()
    return 

# -----------------
# KM3NET FUNCTIONS
# -----------------
def write_K3_total_effective_area(filename,xxs,llt, eps = 1e-2):
    """
    Writes a file "filename" with a table of the total KM3NeT effective area,
    separated among the different contributions (muons, N and T).
    """   
    file = open(filename,'w+')     
    for xs in xxs:
        for lt in llt:
            aeffmu = K3.TotalEffectiveAreaMuon(xs,lt,eps=eps)
            aeffN  = K3.TotalEffectiveAreaN(xs,lt,eps=eps)
            aeffT  = K3.TotalEffectiveAreaT(xs,lt,eps=eps)
            file.write('{0:1.5e},{1:1.5e},{2:1.5e},{3:1.5e},{4:1.5e}\n'.format(xs,lt,aeffmu,aeffN,aeffT))
            print('{0:1.5e},{1:1.5e},{2:1.5e},{3:1.5e},{4:1.5e}\n'.format(xs,lt,aeffmu,aeffN,aeffT))
    file.close()
    return 

def write_K3_avgd_effective_area(filename,xxs,llt, eps = 1e-2):
    """
    Writes a file "filename" with a table of the averaged KM3NeT effective area around the event
    separated among the different contributions (muons, N and T).
    """   
    file = open(filename,'w+')
    for xs in xxs:
        for lt in llt:
            avgd_aeffmu1 = K3.AverageEffectiveAreaMuon(xs,lt,eps = eps)
            file.write('{0:1.5e},{1:1.5e},{2:1.5e}\n'.format(xs,lt,avgd_aeffmu1))
            print('{0:1.5e},{1:1.5e},{2:1.5e}\n'.format(xs,lt,avgd_aeffmu1))
    file.close()
    return


# --------------------------------------------------------- #
#                   RUNNING REGION                          #
# --------------------------------------------------------- #

# We set in which directories to save the files
outdirIC = homedir + 'Data/IceCube/'
outdirK3 = homedir + 'Data/KM3NeT/'
outdirAN = homedir + 'Data/ANITA/'

# We set which parameters to compute areas for
datxs = np.logspace(-30,-35,100)
datlt = np.logspace(-7,1,100)

# ------------------------------------------------------------
# We set whether we want to modify some variables or not
# pars.is_T_absorbed = True

# pars.y_Ttomu = 0.8
# K3.maxEnergy = pars.y_Ttomu*pars.y_NtoT*pars.EnergyDelta

# and so on!
# ------------------------------------------------------------

begin = time.time() # useful to count how much time it takes

# Here follow some examples, better uncomment one at a time :)

# WRITING KM3NeT EFFECTIVE AREA
# filename = outdirK3 + 'EffectiveArea_user.dat'
# write_K3_total_effective_area(filename, datxs, datlt,y=0.5)

# filename = outdirK3 + 'AveragedArea_user.dat'
# write_K3_avgd_effective_area(filename, datxs, datlt, y = 0.5)

# WRITING ICECUBE EFFECTIVE AREA
# ----------------------------------
# filename = outdirIC + 'EffectiveAreas_user.dat'
# write_IC_total_effective_areas(filename, datxs, datlt)

# WRITING ALL ANITA EFFECTIVE AREAS
# ----------------------------------
# filename = outdirAN + 'EffectiveAreas_NoAbsorption.dat'
# write_ANITA_all_effective_areas(filename, datxs, datlt)

print(time.time()-begin)
