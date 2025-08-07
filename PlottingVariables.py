# This function contains different functions and variables useful for plotting.

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import MyUnits as muns

color1 = '#FFB14E'
darkcolor1 = '#E07F00'
color2 = '#EA5F94'
color3 = '#0000FF'

# A gradient of 10 different shades of blue and red
colors =  ['#00429d', '#2c56a0', '#416aa3', '#507ea7', '#5d93ac', '#68a9b1', '#73bfb8', '#7ed4c1', '#89eacf', '#96ffea']
colors2 = ['#670c0c', '#771d1a', '#872b27', '#973934', '#a74743', '#b75551', '#c76360', '#d7726f', '#e8807f', '#f88f8f']
# A loop of different dashings.
dashing = ['-','--','-.',':','-','--','-.',':','-','--','-.',':','-','--','-.',':']

titl_size = 18.
font_size = 14.
tick_size = 13.

figsize = (5,6)
margins = dict(left=0.2, right=0.955,bottom=0.145, top=0.93)
margins_topaxis = dict(left=0.16, right=0.955,bottom=0.11, top=0.88)
margins_twoaxis = dict(left=0.15, right=0.87,bottom=0.11, top=0.88)

cplot_figsize = (5.2,4)
cplot_figsize = (4.5,5.2)
cplot_figsize = (6,7)
cplot_margins = dict(left=0.15, right=0.95,bottom=0.13, top=0.85)
cplot_margins = dict(left=0.17, right=0.85,bottom=0.018, top=0.88)

sigma1 = 2.30
sigma2 = 6.18
sigma3 = 11.83

def add_style(path_to_style):
    plt.style.use(path_to_style)
    matplotlib.rcParams.update({'text.usetex': True})
    matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
    matplotlib.rcParams['figure.facecolor'] = 'white'

# -------------------------------------------------
# CREATING LABELS
# -------------------------------------------------
def sci_not(number):
    # Returns a LaTeX string with the number written in scientific notation.
    pow = int(np.floor(np.log10(number)))
    fac = number/10**pow

    if number == 1.0:
        return r'$1$'
    elif fac == 1.0:
        return r'$10^{{ {power} }}$'.format(power = pow)
    elif np.abs(fac-np.rint(fac)) < 1e-3:
        fac = int(np.rint(fac))
        return r'${factor:d}\times 10^{{ {power} }}$'.format(factor = fac, power = pow)
    else:
        return r'${factor:.2f}\times 10^{{ {power} }}$'.format(factor = fac, power = pow)

def label_sigma(xs):
    return r'$\sigma = \,$'+sci_not(xs)+r'$\,\textrm{cm}^{ 2}$'

def label_tau(lt):
    return r'$\tau = \,$'+sci_not(lt)+r'$\,\textrm{s}$'

def ticks_list(imin,imax):
    ticklist = []
    for i in range(int(imin),int(imax)+1):
        ticklist.append(i)
    return ticklist

def ticks_exp_label(imin,imax):
    ticklist = ticks_list(int(imin),int(imax))
    labels = []
    for i in ticklist:
        if i == 0:
            labels.append(r'$1$')
        else:
            labels.append(r'$10^{{ {} }}$'.format(i))
    return labels

def logticks_list(imin,imax, lims = None):
    ticks = []
    for i in range(int(imin),int(imax)+1):
        x0 = 10**i
        for j in range(2,10):
            if lims == None:
                ticks.append(np.log10(j*x0))
            elif (np.log10(j*x0) >= lims[0]) and (np.log10(j*x0) <= lims[1]):
                # print(np.log10(j*x0),lims[0], lims[1])
                ticks.append(np.log10(j*x0))

    return ticks

def floor_to_int(num):
    # let's take 2.5e-3
    pow = int(np.floor(np.log10(num))) # this gives -3
    fac = num/10**pow # this gives 2.5
    my_floor = np.floor(fac) # this gives 2
    return my_floor*10**pow # this gives 2e-3

def ceil_to_int(num):
    pow = int(np.floor(np.log10(num))) # this gives -3
    fac = num/10**pow # this gives 2.5
    my_ceil = np.ceil(fac) # this gives 3
    return my_ceil*10**pow # this gives 3e-3


# -----------------------------------------------------------
# CREATING AXES
# -----------------------------------------------------------

def set_colorbar_ticks(cbar, imin, imax, where = 'vertical'):
    if where == 'vertical':
        cbar.ax.set_xticks(ticks_list(np.ceil(imin),np.floor(imax)))
        cbar.ax.set_xticklabels(ticks_exp_label(np.ceil(imin),np.floor(imax)), fontsize = tick_size)
        cbar.ax.set_xticks(logticks_list(np.floor(imin),np.ceil(imax)-1,(imin,imax)), labels = None, minor = True)
    else:
        cbar.ax.set_yticks(ticks_list(np.ceil(imin),np.floor(imax)))
        cbar.ax.set_yticklabels(ticks_exp_label(np.ceil(imin),np.floor(imax)), fontsize = tick_size)
        cbar.ax.set_yticks(logticks_list(np.floor(imin),np.ceil(imax)-1,(imin,imax)), labels = None, minor = True)
    return 

def set_colorbar(fig,ax, conts, imin, imax, lab = 'area', loc = 'bottom'):
    if lab == 'area':
        lab = r'Effective area $A_{\footnotesize\textrm{eff}}\ (\textrm{km}^2)$'
    elif lab == 'flux':
        lab = r'Flux $\Phi \ (\textrm{km}^{-2}\textrm{day}^{-1})$'
    elif lab == 'nume':
        lab = r'Expected number of events in IceCube'
    elif lab == 'prob':
        lab = r'Probability of zero events in IceCube'


    cbar = fig.colorbar(conts, pad = 0.16, label = lab, ax = ax, location = loc, shrink = 1.25)#, orientation = 'horizontal')
    cbar.ax.set_xticks(ticks_list(np.ceil(imin),np.floor(imax)))
    cbar.ax.set_xticklabels(ticks_exp_label(np.ceil(imin),np.floor(imax)), fontsize = tick_size)
    cbar.ax.set_xticks(logticks_list(np.floor(imin),np.ceil(imax)-1,(imin,imax)), labels = None, minor = True)
    if loc == 'right':
        cbar.ax.set_ylabel(lab, fontsize = font_size)
    if loc == 'bottom':
        cbar.ax.set_xlabel(lab, fontsize = font_size)
    return

def create_xslt_axis(ax,xsmin,xsmax,ltmin,ltmax):
    xsinf  = np.log10(xsmin*1e32)
    xssup  = np.log10(xsmax*1e32)
    ltinf  = np.log10(ltmin)
    ltsup  = np.log10(ltmax)
    ixsmax = int(np.floor(xssup)) + 1
    ixsmin = int(np.floor(xsinf)) 
    iltmax = int(np.floor(ltsup)) + 1
    iltmin = int(np.floor(ltinf)) 

    ax.set_yticks(ticks_list(ixsmin,ixsmax), labels = ticks_exp_label(ixsmin,ixsmax), fontsize = tick_size)
    ax.set_xticks(ticks_list(iltmin,iltmax), labels = ticks_exp_label(iltmin,iltmax), fontsize = tick_size)
    ax.set_yticks(logticks_list(ixsmin-1,ixsmax,(xsinf,xssup)), labels = None, minor = True)
    ax.set_xticks(logticks_list(iltmin-1,iltmax,(ltinf,ltsup)), labels = None, minor = True)
    ax.set_ylim([xsinf,xssup])
    ax.set_xlim([ltinf,ltsup])
    ax.tick_params(axis = 'both', which = 'minor', width = 0.5)

    ax.set_ylabel(r'$\sigma\ (10^{-32}\textrm{ cm}^2)$', fontsize = font_size)
    ax.set_xlabel(r'$\tau\ (\textrm{s})$', fontsize = font_size)
    return

def create_mfp_axis(ax, xsmin,xsmax):
    ax_mfp = ax.twinx()
    mfpmax = np.log10(muns.MeanFreePath(xsmax)/muns.km)
    mfpmin = np.log10(muns.MeanFreePath(xsmin)/muns.km)
    ax_mfp.set_ylim(mfpmin, mfpmax)
    imax = int(np.floor(mfpmin)) 
    imin = int(np.floor(mfpmax)) + 1
    
    ax_mfp.set_yticks(ticks_list(imin, imax), labels = ticks_exp_label(imin,imax), fontsize = tick_size)
    ax_mfp.set_yticks(logticks_list(imin-1,imax,(mfpmax,mfpmin)), labels = None, minor = True)
    ax_mfp.set_ylabel(r"$\lambda_{\textrm{crust}}\ (\mathrm{km}) $", fontsize = font_size)
    ax_mfp.tick_params(axis = 'y', which = 'minor', width = 0.5)
    return

def create_decaylength_axis(ax,ltmin,ltmax):
    ax_dl = ax.twiny()
    dlmin = np.log10(muns.DecayLength(ltmin)/muns.km)
    dlmax = np.log10(muns.DecayLength(ltmax)/muns.km)
    ax_dl.set_xlim(dlmin,dlmax)
    imin = int(np.floor(dlmin)) + 1
    imax = int(np.floor(dlmax))


    ax_dl.set_xticks(ticks_list(imin,imax),ticks_exp_label(imin,imax), fontsize = tick_size)
    ax_dl.set_xticks(logticks_list(imin-1,imax,(dlmin,dlmax)), labels = None, minor = True)
    ax_dl.set_xlabel(r"$c\tau\ (\mathrm{km}) $", labelpad = 10.0, fontsize = font_size)
    ax_dl.tick_params(axis = 'x', which = 'minor', width = 0.5)
    return

def create_philt_axis(ax,phimin,phimax,ltmin,ltmax):
    ax.set_yticks(ticks_list(phimin,phimax), labels = ticks_exp_label(phimin,phimax), fontsize = tick_size)
    ax.set_xticks(ticks_list(ltmin,ltmax), labels = ticks_exp_label(ltmin,ltmax), fontsize = tick_size)
    ax.set_yticks(logticks_list(phimin,phimax), labels = None, minor = True)
    ax.set_xticks(logticks_list(ltmin,ltmax), labels = None, minor = True)
    ax.set_ylim([phimin,phimax])
    ax.set_xlim([ltmin,ltmax])
    ax.tick_params(axis = 'both', which = 'minor', width = 0.5)

    ax.set_ylabel(r'Flux $\Phi\ (\textrm{km}^{-2}\textrm{ day}^{-1})$', fontsize = font_size)
    ax.set_xlabel(r'$\tau\ (\textrm{s})$', fontsize = font_size)
    return