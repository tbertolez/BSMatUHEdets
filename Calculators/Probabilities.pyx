import MyUnits as muns
import Parameters as pars
import numpy as np
# cimport numpy as np 
from libc.math cimport log10, exp, log, pow
from libc.stdio cimport printf 

#####################################################
#    PROBABILITIES FOR THE PRIMARY PARTICLE         #
#####################################################

cdef double ProbN_AbsorptionT(double [:,:] lay, double efin, double einiN, double xs, double lt):
    """
    Implements equation (B.17) (but there are erratums there haha).
    N is produced from T interaction (via xs) -one regeneration-.
    T can be absorbed by the cross-section sigma, losing all its energy and not producing "anything".
    """
    cdef double ctau, lam, d, prob, dx, rho, sum_A, sum_B, prod_all
    cdef int m = lay.shape[0]
    cdef double[:] prods_pre  = np.empty(m, dtype=np.float64)
    cdef double[:] prods_post = np.empty(m, dtype=np.float64)
    cdef double[:,:] prods_mid = np.ones((m,m), dtype=np.float64)
    cdef double[:,:] invlay
    cdef int i, j

    if ((einiN != efin) and (einiN*pars.y_NtoT*pars.y_TtoN != efin)):
        return 0.0
    
    #m = lay.shape[0]
    ctau = muns.c*lt

    #if len(lay.shape) == 1:
    if (lay[0,0] == lay[1,0]):
        # only one layer (aka only water)
        lam = muns.ProtonMass/xs/lay[0,-1]
        d = lay[0,2]

        if einiN==efin:
            return exp(-d/lam)
        else: # means einiN*pars.y_NtoT*pars.y_TtoN == efin
            return exp(-d/lam)*(pow(ctau/lam,2.)*(d/ctau+exp(-d/ctau)-1.))

    # if more than one layer
    # This implements the probability of surviving all layers (zero order term)
    prod_all = 1.
    for i in range(m):
        lam = muns.ProtonMass/xs/lay[i, 3]
        dx = lay[i,2]
        prod_all = prod_all*exp(-dx/lam)

    if einiN==efin:
        return prod_all
    
    elif einiN*pars.y_NtoT*pars.y_TtoN == efin:
        # This implements the probability of having survived all layers previous to layer i
        prods_pre[0] = 1.0
        for i in range(1, m):
            lam = muns.ProtonMass/xs/lay[i-1, 3]
            prods_pre[i] = prods_pre[i-1] * exp(-lay[i-1, 2] / lam)

        # This implements the probability of surviving all layers after layer i
        prods_post[0] = 1.0
        invlay = np.flip(lay, axis = 0)
        for i in range(1, m):
            lam = muns.ProtonMass/xs/invlay[i-1, 3]
            prods_post[i] = prods_post[i-1] * exp(-lay[i-1, 2] / lam)
        prods_post = prods_post[::-1]

        # This implements the probability of surviving all layers between i and j
        for i in range(m):
            for j in range(i+2, m):
                dx = lay[j-1,2]
                rho = lay[j-1,3]
                lam = muns.ProtonMass/xs/rho
                prods_mid[i,j] = prods_mid[i,j-1]*exp(-dx/lam-dx/ctau) # there was a dx/ctau missing here

        # Computes the first term (T and N production happen on the same layer)
        sum_A = 0.
        for i in range(m):
            dx  = lay[i,2]
            rho = lay[i,3]
            lam = muns.ProtonMass/xs/rho
            
            sum_A = sum_A + (prods_pre[i]*
                        (exp(-dx/lam)*pow(ctau/lam,2.)*(dx/ctau+exp(-dx/ctau)-1))*
                        prods_post[i]
                        ) 
        
        # Computes the second term (T and N production happen on different layers)
        sum_B = 0.
        for i in range(m):
            dxi = lay[i,2]
            lami = muns.ProtonMass/xs/lay[i,3]
            for j in range(i+1,m):
                dxj = lay[j,2]
                lamj = muns.ProtonMass/xs/lay[j,3]
                
                sum_B = sum_B + (prods_pre[i]*
                                    (ctau/lami*exp(-dxi/lami)*(1-exp(-dxi/ctau)))*
                                    prods_mid[i,j]*
                                    (ctau/lamj*exp(-dxj/lamj)*(1-exp(-dxj/ctau)))*
                                    prods_post[j]
                                )  

        return (sum_A+sum_B)


cdef double ProbN_NoAbsorption(double [:,:] lay, double efin, double einiN, double xs, double lt):
    """
    Implements equation (B.17) (but there are erratums there haha).
    T cannot be absorbed by the cross-section sigma
    """
    cdef double ctau, lam, d, prob, dx, rho, sum_A, sum_B, prod_all
    cdef int m = lay.shape[0]
    cdef int i, j

    if (einiN != efin):
        return 0.0
    
    ctau = muns.c*lt

    if (lay[0,0] == lay[1,0]):
        # only one layer (aka only water)
        lam = muns.ProtonMass/xs/lay[0,-1]
        d = lay[0,2]
        return exp(-d/lam)

    # if more than one layer
    # This implements the probability of surviving all layers (zero order term)
    prod_all = 1.
    for i in range(m):
        lam = muns.ProtonMass/xs/lay[i, 3]
        dx = lay[i,2]
        prod_all = prod_all*exp(-dx/lam)

    return prod_all

cpdef double ProbN(double [:,:] lay, double efin, double einiN, double xs, double lt, bint is_T_absorbed):
    if is_T_absorbed:
        return ProbN_AbsorptionT(lay,efin,einiN,xs,lt)
    else:
        return ProbN_NoAbsorption(lay,efin,einiN,xs,lt)


#####################################################
#  PROBABILITIES FOR THE SECONDARY PARTICLE         #
#####################################################

cpdef double ProbDecayIn(double lt, double r):
    # probability of decaying inside the detector
    return 1 - exp(-r/(muns.c*lt))

cpdef double ProbT_AbsorptionT(double [:,:] lay, double efin, double einiN, double xs, double lt):
    """
    Implements equation (B.20), T particles can get absorbed by cross-section xs.
    Input:
    azim,elev, xs, lt (float): azimuthal angle, elevation angle, cross-section and lifetime.
    Output (float): the probability of exit.
    """
    cdef double ctau, lam, d, prob, dx, rho, suma, Radius, prod_all
    cdef int m = lay.shape[0]
    cdef double[:] prods_post = np.empty(m, dtype=np.float64)
    cdef double[:,:] invlay
    cdef int i

    if efin!=einiN*pars.y_NtoT:
        return 0.0

    ctau = muns.c*lt

    if (lay[0,0] == lay[1,0]):
        # only one layer (aka only water)
        lam = muns.ProtonMass/xs/lay[0,-1]
        d = lay[0,2]
        return exp(-d/lam)*ctau/lam*(1.-exp(-d/ctau))

    # Probability of a N/T particle surviving all layers without interacting
    prod_all = 1.
    for i in range(m):
        lam = muns.ProtonMass/xs/lay[i, 3]
        prod_all = prod_all*exp(-lay[i,2]/lam)
    
    prods_post[0] = 1.0
    invlay = np.flip(lay, axis = 0)
    for i in range(1, m):
        prods_post[i] = prods_post[i-1] * exp(-lay[i-1, 2] / ctau) # lambda is included in prod_all
    prods_post = prods_post[::-1]

    suma = 0.
    for i in range(m):
        dx  = lay[i,2]
        rho = lay[i,3]
        
        lam  = muns.ProtonMass/xs/rho
        suma = suma + ctau/lam*(1-exp(-dx/ctau))*prods_post[i]

    return suma*prod_all


cpdef double ProbT_NoAbsorption(double [:,:] lay, double efin, double einiN, double xs, double lt):
    """
    Implements equation (B.20), T particles do not get absorbed by cross-section xs.
    Input:
    azim,elev, xs, lt (float): azimuthal angle, elevation angle, cross-section and lifetime.
    Output (float): the probability of exit.
    """
    cdef double ctau, lam, d, prob, dx, rho, suma, prod_all, ctl
    cdef int m = lay.shape[0]
    cdef double[:] prods_pre = np.empty(m, dtype=np.float64)
    cdef double[:] prods_post = np.empty(m, dtype=np.float64)
    cdef double[:,:] invlay
    cdef int i

    if efin!=einiN*pars.y_NtoT:
        return 0.0

    ctau = muns.c*lt

    if (lay[0,0] == lay[1,0]):
        # only one layer (aka only water)
        lam = muns.ProtonMass/xs/lay[0,-1]
        d = lay[0,2]
        return ctau/(ctau-lam)*(exp(-d/ctau)-exp(-d/lam))

    prods_pre[0] = 1.0
    for i in range(1, m):
        lam = muns.ProtonMass/xs/lay[i-1, 3]
        prods_pre[i] = prods_pre[i-1] * exp(-lay[i-1, 2] / lam)
    
    prods_post[0] = 1.0
    invlay = np.flip(lay, axis = 0)
    for i in range(1, m):
        prods_post[i] = prods_post[i-1] * exp(-lay[i-1, 2] / ctau)
    prods_post = prods_post[::-1]

    # The total probability
    suma = 0.
    for i in range(m):
        dx  = lay[i,2]
        rho = lay[i,3]
        
        lam  = muns.ProtonMass/xs/rho
        ctl = 1./(1./lam-1./ctau)
        suma += ctl/lam*(exp(-dx/ctau)-exp(-dx/lam))*prods_post[i]*prods_pre[i]

    return suma


cpdef double ProbT(double [:,:] lay, double efin, double einiN, double xs, double lt, bint is_T_absorbed):
    if is_T_absorbed:
        return ProbT_AbsorptionT(lay,efin,einiN,xs,lt)
    else:
        return ProbT_NoAbsorption(lay,efin,einiN,xs,lt)


#####################################################
#            PROBABILITIES FOR THE MUON             #
#####################################################

#cdef double dist_çmu_10GeV(double emu):
#    eps = 500*muns.GeV
#    return 1.8e5*muns.cm*(np.log((emu+eps)/(10*muns.PeV+eps)))**0.72


# Probability for a muon above 10PeV
#######################################


cpdef double ProbMuon_dmin_AbsorptionT(double [:,:] lay, double eini, double xs, double lt):
    # Implements the probability for a muon of arriving at the detector with energy larger than 10 PeV, if T can be absorbed by Earth.
    cdef double ctau, dmin, lam, d, prob, x0, x1, dx, rho, termA, termB, termB2k, prob_smalld, prob_bigd,eps, prod_pre,rhor,rhow,dw
    cdef int n = lay.shape[0]
    # cdef double[:] prods_pre = np.empty(n, dtype=np.float64)
    cdef double[:, :] prods_mid = np.ones((n, n), dtype=np.float64)

    cdef int i, j, k, m
    ctau = muns.c*lt
    eps = 500*muns.GeV
    # from Gaisser-Resconi book, this is the distance travelled by a muon from eini to 10PeV
    dmin = 2.5e5*log((eini+eps)/(10*muns.PeV+eps))*muns.cm # we integrate until 10PeV, distance in cm of water

    if (lay[0,0] == lay[1,0]):
        # only one layer (i.e. only water)
        lam = muns.ProtonMass/xs/lay[0,-1]
        ctl = 1./(1./ctau+1./lam)
        d = lay[0,2]
        if d > dmin:
            prob_bigd  = ctau*(exp(-d/ctl)-exp(-d/lam)-exp(-(d-dmin)/ctl))
            prob_bigd += (ctau+lam)*exp(-(d-dmin)/lam)-lam*exp(-d/lam)
            prob_bigd /= ctau+lam
            return prob_bigd
        else:
            prob_smalld  = lam+ctau*exp(-d/ctl)-(ctau+lam)*exp(-d/lam)
            prob_smalld /= ctau+lam
            return prob_smalld

    d    = lay[-1,1] # the total length traveled
    dw   = lay[-1,2] # the distance from the detector to the end of the last layer of water
    rhow = lay[-1,3] # density of last layer (of water)
    rhor = lay[-2,3] # density of last to last layer (of rock)
    efin = 10*muns.PeV

    if (dmin > dw):
        dmin = dw + rhow/rhor*(dmin-dw)
     
    # Now we find in which layer is dmin located
    m = n+1
    for i in range(n):
        x0, x1, dx, rho = lay[n-i-1] # since dmu is usually small, the layer will be one of the last ones, let's begin searching there
        if ((x0 < d-dmin) and (x1 > d-dmin)):
            m = n-i-1 # m marks the layer where the muon must be created
            break
    if m == n+1:
        print("No layer found, this is a problem!")

    # Now we begin to compute the probability
    if m == n-1:
        # the muon is created in the last layer, this simplifies the formula

        # term A: the interaction and the decay happen in the same (last) layer
        prod_pre = 1.0
        for i in range(m): # for prod_pre, the last layer does not count
            lam = muns.ProtonMass/xs/lay[i, 3]
            prod_pre *= exp(-lay[i, 2] / lam)
        
        x0, x1, dx, rho = lay[m]
        lam = muns.ProtonMass/xs/rho
        ctl = 1./(1./ctau+1./lam)
        termA  = ctau/(ctau+lam)*(exp(-(d-x0)/ctl)-exp(-(d-dmin-x0)/ctl))
        termA += exp(-(d-dmin-x0)/lam)-exp(-(d-x0)/lam)
        termA *= prod_pre
        
        # term B: the interaction happens in a layer i before the last one, the decay in m.
        for i in range(m+1): # doing a two-loop is not very efficient, but is readable
            for j in range(i+2, m+1):
                dx  = lay[j-1,2]
                rho = lay[j-1,3]
                lam = muns.ProtonMass/xs/rho
                prods_mid[i,j] = prods_mid[i,j-1]*np.exp(-dx/ctau)

        termB = 0.0
        for i in range(m): # the last layer is not counted, is where decay happens
            x0, x1, dx, rho = lay[i]
            lam = muns.ProtonMass/xs/rho
            termB += (1.-exp(-dx/ctau))*prods_mid[i,m]*ctau/lam

        x0, x1, dx, rho = lay[m]
        lam = muns.ProtonMass/xs/rho
        ctl = 1./(1./ctau+1./lam)
        termB *= prod_pre*lam/(ctau+lam)*(exp(-(d-dmin-x0)/ctl)-exp(-(d-x0)/ctl))

        return termA+termB
    
    else:
        # Muon is not created in the last layer

        prod_pre = 1.0
        for i in range(m): # for prod_pre, the last layer does not count
            lam = muns.ProtonMass/xs/lay[i, 3]
            prod_pre *= exp(-lay[i, 2] / lam)

        for i in range(n): # doing a two-loop is not very efficient, but readable
            for j in range(i+2, n):
                dx  = lay[j-1,2]
                rho = lay[j-1,3]
                lam = muns.ProtonMass/xs/rho
                prods_mid[i,j] = prods_mid[i,j-1]*np.exp(-dx/ctau)
        
        # 1. term corresponding to muon creation in layer m, after dmin
        # --------------------------------------------------------------
        x0, x1, dx, rho = lay[m]
        lam = muns.ProtonMass/xs/rho
        ctl = 1./(1./ctau+1./lam)
        termA  = -exp(-dx/lam)+exp(-(d-x0-dmin)/lam)
        termA += -ctau/(ctau+lam)*(-exp(-dx/ctl)+exp(-(d-x0-dmin)/ctl))
        termA *= prod_pre

        termB  = 0.0
        for i in range(m):
            x0, x1, dx, rho = lay[i]
            lam = muns.ProtonMass/xs/rho
            termB += (1.-exp(-dx/ctau))*prods_mid[i,m]*ctau/lam#*exp(-dx/lam) # this exp(-dx/lam) is double-counted with the prod_pre

        x0, x1, dx, rho = lay[m]
        lam = muns.ProtonMass/xs/rho
        ctl = 1./(1./ctau+1./lam)
        termB *= prod_pre*lam/(ctau+lam)*(exp(-(d-x0-dmin)/ctl)-exp(-dx/ctl))

        # 2. terms corresponding to muon creation in layers between m and n (included)
        # ----------------------------------------------------------------------------
        for k in range(m+1,n): # k is the layer where the muon is produced
            prod_pre = 1.0
            for i in range(k): # the last layer does not count
                lam = muns.ProtonMass/xs/lay[i, 3]
                prod_pre *= exp(-lay[i, 2] / lam)
            
            x0, x1, dx, rho = lay[k]
            lam = muns.ProtonMass/xs/rho
            ctl = 1./(1./ctau+1./lam)
            termA += prod_pre*(1-exp(-dx/lam)-ctau/(ctau+lam)*(1.-np.exp(-dx/ctl)))

            termB2k = 0.0
            for i in range(k):
                x0, x1, dx, rho = lay[i]
                lam = muns.ProtonMass/xs/rho
                termB2k += (1.-exp(-dx/ctau))*prods_mid[i,k]*ctau/lam#*exp(-dx/lam) # this exp(-dx/lam) is double-counted with the prod_pre

            x0, x1, dx, rho = lay[k]
            lam = muns.ProtonMass/xs/rho
            ctl = 1./(1./ctau+1./lam)
            termB2k *= prod_pre*lam/(ctau+lam)*(1.-exp(-dx/ctl))
            termB += termB2k

        # Finally done, this is a hard formula!!!!!        
        return termA+termB


cpdef double ProbMuon_dmin_NoAbs(double [:,:] lay, double eini, double xs, double lt):
    # Implements the probability for a muon of arriving at the detector with energy larger than 10 PeV, if T cannot be absorbed by Earth.
    cdef double ctau, dmin, dmin1, lam, d, prob, x0, x1, dx, rho, term1, term2, prob_smalld, prob_bigd,eps
    cdef int n = lay.shape[0]
    cdef double[:] prods_pre = np.empty(n, dtype=np.float64)
    cdef int i
    ctau = muns.c*lt
    eps = 500*muns.GeV
    dmin = 2.5e5*log((eini+eps)/(10*muns.PeV+eps))*muns.cm # we integrate until 10PeV, distance in cm of water

    if (lay[0,0] == lay[1,0]):
        # only one layer (i.e. only water)
        lam = muns.ProtonMass/xs/lay[0,-1]
        d = lay[0,2]

        if d > dmin:
            prob_bigd  = ctau*(exp(-(d-dmin)/ctau)-exp(-d/ctau))-lam*(exp(-(d-dmin)/lam)-exp(-d/lam))
            prob_bigd /= ctau-lam
            return prob_bigd
        else:
            prob_smalld  = ctau*(1.-exp(-d/ctau))+lam*(exp(-d/lam)-1.)
            prob_smalld /= ctau-lam
            return prob_smalld

    # if more than one layer
    prods_pre[0] = 1.0
    for i in range(1, n):
        lam = muns.ProtonMass/xs/lay[i-1, 3]
        prods_pre[i] = prods_pre[i-1] * exp(-lay[i-1, 2] / lam)

    d = lay[-1,1] # the total length traveled
    dw   = lay[-1,2] # the distance from the detector to the end of the last layer of water
    rhow = lay[-1,3] # density of last layer (of water)
    rhor = lay[-2,3] # density of last to last layer (of rock)

    if (dmin > dw):
        dmin = dw + rhow/rhor*(dmin-dw)

    # Now we begin to compute the probability
    prob = 0.0
    for i in range(n):
        x0, x1, dx, rho = lay[i]
        lam = muns.ProtonMass/xs/rho

        if (d > dmin):
            term1 = (exp(-(d-x1)/ctau)*ctau + lam - ctau)*exp(-dx/lam)
            if (d-dmin-x0 > 0.0) and (dmin + x1 - d < 0.0):
                term1 += exp(-dx/lam)*(ctau-lam-ctau*exp(-(d-dmin-x1)/ctau))

            term2 = (-ctau*exp(-(d-x0)/ctau)+ ctau - lam)
            if (d-dmin-x0 > 0.0):
                term2 += (ctau*(exp(-(d-dmin-x0)/ctau)-1.)+lam)
                if (-d+dmin+x1>0.0):
                    term2 -= lam*exp(-(d-dmin-x0)/lam)

            prob += (term1+term2)/(ctau-lam)*prods_pre[i]
        else:
            prob_smalld = 1.-exp(-dx/lam)+ctau/(ctau-lam)*(exp(-(d-x1)/ctau-dx/lam)-exp(-(d-x0)/ctau))
            prob_smalld *= prods_pre[i]
            prob += prob_smalld
    
    # This one was a bit easier :)
    return prob


cpdef double ProbMuon_dmin(double [:,:] lay, double eini, double xs, double lt, bint is_T_absorbed):
    # Implements the probability for a muon of arriving at the detector with energy larger than 10 PeV, if T cannot be absorbed by Earth.
    if is_T_absorbed:
        return ProbMuon_dmin_AbsorptionT(lay,eini,xs,lt)
    else:
        return ProbMuon_dmin_NoAbs(lay,eini,xs,lt)


# The previous functions are the ones which are used in the real analysis.
# Extra functions are provided here for reference. Use with care!
# ---------------------------------------------------------------------

#  ------------------------------------------------
#   GIVEN DISTANCE PROBABILITIES
#  ------------------------------------------------


cdef double ProbMuonAtd_NoAbs(double[:,:] lay, double dmu, double xs, double lt):
    # Probability of a muon being created at a distance dmu from the detector, with T not absorbed
    cdef double ctau, rhow, rhor, lam, dmu, d, prob, x0, x1, dx, rho, term
    cdef int n = lay.shape[0]
    cdef double[:] prods_pre = np.empty(n, dtype=np.float64)
    cdef int i
    ctau = muns.c*lt

    if (lay[0,0] == lay[1,0]):
        # only one layer (aka only water)
        rhow = lay[0,-1]
        lam = muns.ProtonMass/xs/rhow
        d = lay[0,2]
        if d < dmu:
            return 0.0
        return 1./(ctau-lam)*(exp(-(d-dmu)/ctau)-exp(-(d-dmu)/lam))

    # if more than one layer
    d = lay[-1,1] # the total length traveled

    if d < dmu:
        return 0.0

    prods_pre[0] = 1.0
    for i in range(1, n):
        lam = muns.ProtonMass/xs/lay[i-1, 3]
        prods_pre[i] = prods_pre[i-1] * exp(-lay[i-1, 2] / lam)

    prob = 0.0
    for i in range(n):
        x0, x1, dx, rho = lay[i]
        prod_pre = prods_pre[i]
        lam = muns.ProtonMass/xs/rho
        if (d-dmu-x0>0):
            if (x1>d-dmu):
                term = (exp(-(dx+d-dmu-x1)/ctau)-exp(-(d-dmu-x0)/lam))
            else:
                term  = exp(-(d-dmu-x1)/ctau)*(exp(-dx/ctau)-exp(-dx/lam)) # this doesn't overflow

            term *= 1./(ctau-lam)*prod_pre
            prob += term

    return prob

cdef double ProbMuonAtd_TAbsorption(double[:,:] lay, double dmu, double xs, double lt):
    # Probability of a muon being created at a distance dmu from the detector, with T can absorbed
    cdef double ctau, rhow, rhor, lam, dmu, d, x0, x1, dx, rho, termA, termB, prod_pre
    cdef int n = lay.shape[0]
    cdef double[:, :] prods_mid = np.ones((n, n), dtype=np.float64)
    cdef int i, j, m, k
    ctau = muns.c*lt
    
    if (lay[0,0] == lay[1,0]):
        # only one layer (aka only water)
        rhow = lay[0,-1]
        lam = muns.ProtonMass/xs/rhow
        d = lay[0,2]
        if d < dmu:
            return 0.0
        return 1./lam*exp(-(d-dmu)/lam)*(1.-exp(-(d-dmu)/ctau))
    
    d = lay[-1,1] # the total length traveled
    if d<dmu:
        return 0.0
    
    m = n+1
    for i in range(n):
        x0, x1, dx, rho = lay[n-i-1] # since dmu is usually small, the layer will be one of the last ones, let's begin searching there
        if ((x0 < d-dmu) and (x1 > d-dmu)):
            m = n-i-1 # m marks the layer where the muon must be created
            break
    if m == n+1:
        print("No layer found, this is a problem!")

    prod_pre = 1.0 # actually this is prod_all
    for i in range(m):
        lam = muns.ProtonMass/xs/lay[i, 3]
        prod_pre *= exp(-lay[i, 2] / lam)

    x0, x1, dx, rho = lay[m]
    lam = muns.ProtonMass/xs/rho
    termA = prod_pre/lam*exp(-(d-dmu-x0)/lam)*(1.-exp(-(d-dmu-x0)/ctau))
    
    for i in range(m+1):
        for j in range(i+2, m+1):
            dx  = lay[j-1,2]
            rho = lay[j-1,3]
            lam = muns.ProtonMass/xs/rho
            prods_mid[i,j] = prods_mid[i,j-1]*np.exp(-dx/ctau)

    termB = 0.0
    for i in range(m):
        x0, x1, dx, rho = lay[i]
        lam = muns.ProtonMass/xs/rho
        termB += exp(-dx/lam)*(1.-exp(-dx/ctau))*prods_mid[i,m]/lam
    
    x0, x1, dx, rho = lay[m]
    lam = muns.ProtonMass/xs/rho
    termB *= prod_pre*exp(-(d-dmu-x0)*(1./ctau+1./lam))

    return termA + termB

cpdef double ProbMuonAtd(double [:,:] lay, double dmu, double xs, double lt, bint is_T_absorbed):
    # Probability of a muon being created at a distance dmu from the detector
    if is_T_absorbed:
        return ProbMuonAtd_AbsorptionT(lay,dmu,xs,lt)
    else:
        return ProbMuonAtd_NoAbs(lay,dmu,xs,lt)


#  ------------------------------------------------
#   ENERGY-DEPENDENT PROBABILITIES
#  ------------------------------------------------

cdef double mub_rock(double emu):
    return 3.55e-6*pow(log10(emu),1/6.5) # heuristic fit, more or less valid for E > 1e4 GeV

cdef double mub_water(double emu):
    return 2.6e-6*pow(log10(emu),1/5.5) # heuristic fit, more or less valid for E > 1e4 GeV

cdef double ProbMuon_NoAbs(double [:,:] lay, double efin, double einiN, double xs, double lt):
    # Probability of a muon arriving at the detector with energy efin, T cannot be absorbed
    cdef double ctau, einimu, rhow, rhor, lam, dmu, d, prob, x0, x1, dx, rho, factor, term
    cdef int n = lay.shape[0]
    cdef double[:] prods_pre = np.empty(n, dtype=np.float64)
    cdef int i
    ctau = muns.c*lt
    einimu = einiN*pars.y_NtoT*pars.y_Ttomu
    
    if einimu < efin:
        return 0.0

    if (lay[0,0] == lay[1,0]):
        # only one layer (aka only water)
        
        rhow = lay[0,-1]
        lam = muns.ProtonMass/xs/rhow
        dmu = 1./mub_water(einimu)/rhow*log(einimu/efin)
        d = lay[0,2]
        if d < dmu:
            return 0.0
        return 1./(ctau-lam)*(exp(-(d-dmu)/ctau)-exp(-(d-dmu)/lam))/efin/(mub_water(einimu)*rhow)

    # if more than one layer
    d = lay[-1,1] # the total length traveled
    dw = lay[-1,2] # the distance from the detector to the end of the last layer of water
    rhow = lay[-1,3] # density of last layer (of water)
    rhor = lay[-2,3] # density of last to last layer (of rock)

    # now we compute the distance travelled by the muon
    # perhaps it might be better to extend this to three layers!!! very specific weird things can happen and we can use rock when we should use water...
    if (log(einimu/efin)/mub_water(einimu)/rhow > dw):
        dmu  = (dw + 1/mub_rock(einimu)/rhor*(log(einimu/efin)-mub_water(einimu)*rhow*dw))
    else:
        dmu  = (1./mub_water(einimu)/rhow*log(einimu/efin))

    if d<dmu:
        return 0.0

    prods_pre[0] = 1.0
    for i in range(1, n):
        lam = muns.ProtonMass/xs/lay[i-1, 3]
        prods_pre[i] = prods_pre[i-1] * exp(-lay[i-1, 2] / lam)

    prob = 0.0
    for i in range(n):
        #print(lay)
        x0, x1, dx, rho = lay[i]
        prod_pre = prods_pre[i]
        lam = muns.ProtonMass/xs/rho
        if (d-dmu-x0>0):
            if (x1>d-dmu):
                # HERE x1 > d-dmu, so this term is positive and very big.
                # You need to factorize it and compute it together with term1
                # term = exp(-(d-dmu-x1)/ctau)*(exp(-dx/ctau)-exp(-dx/lam)) + (exp(-(d-dmu-x1)/ctau-dx/lam)-exp(-(d-dmu-x0)/lam))
                term = exp(-(d-dmu-x1)/ctau)*(exp(-dx/ctau)-exp(-(d-dmu-x0)/lam+(d-dmu-x1)/ctau))
                
                # term  = exp(-(d-dmu-x1)/ctau)*(exp(-dx/ctau)-exp(-dx/lam))
                # term += (exp(-(d-dmu-x1)/ctau-dx/lam)-exp(-(d-dmu-x0)/lam))
                #print("Term2: {:.5e}".format(term))
            else:
                term  = exp(-(d-dmu-x1)/ctau)*(exp(-dx/ctau)-exp(-dx/lam)) # this doesn't overflow
                #print("Term1: {:.5e}".format(term))

            #print("Prodpre: {:.5e}".format(prod_pre))
            term *= 1./(ctau-lam)*prod_pre
            prob += term

    if (log(einimu/efin)/mub_water(einimu)/rhow > dw):
        factor = 1./(mub_rock(einimu)*rhor)
    else:
        factor = 1./(mub_water(einimu)*rhow)

    return prob/efin*factor

cdef double ProbMuon_AbsorptionT(double [:,:] lay, double efin, double einiN, double xs, double lt):
    # Probability of a muon arriving at the detector with energy efin, T cannot be absorbed
    cdef double ctau, einimu, rhow, rhor, lam, dmu, d, x0, x1, dx, rho, termA, termB, prod_pre, factor
    cdef int n = lay.shape[0]
    cdef double[:, :] prods_mid = np.ones((n, n), dtype=np.float64)
    cdef int i, j, m, k

    ctau = muns.c*lt
    einimu = einiN*pars.y_NtoT*pars.y_Ttomu

    if einimu <= efin:
        return 0.0
    
    if (lay[0,0] == lay[1,0]):
        # only one layer (aka only water)
        rhow = lay[0,-1]
        lam = muns.ProtonMass/xs/rhow
        dmu = 1./mub_water(einimu)/rhow*log(einimu/efin)
        d = lay[0,2]
        if d < dmu:
            return 0.0
        return 1./lam*exp(-(d-dmu)/lam)*(1.-exp(-(d-dmu)/ctau))/efin/(mub_water(einimu)*rhow)
    
    d = lay[-1,1] # the total length traveled
    dw = lay[-1,2] # the distance from the detector to the end of the last layer of water
    rhow = lay[-1,3] # density of last layer (of water)
    rhor = lay[-2,3] # density of last to last layer (of rock)

    # now we compute the distance travelled by the muon
    # perhaps it might be better to extend this to three layers!!! very specific weird things can happen and we can use rock when we should use water...
    if (log(einimu/efin)/mub_water(einimu)/rhow > dw):
        dmu  = (dw + 1/mub_rock(einimu)/rhor*(log(einimu/efin)-mub_water(einimu)*rhow*dw))
    else:
        dmu  = (1./mub_water(einimu)/rhow*log(einimu/efin))
    
    if d<dmu:
        return 0.0
    
    m = n+1
    for i in range(n):
        x0, x1, dx, rho = lay[n-i-1] # since dmu is usually small, the layer will be one of the last ones, let's begin searching there
        if ((x0 < d-dmu) and (x1 > d-dmu)):
            m = n-i-1 # m marks the layer where the muon must be created
            break
    if m == n+1:
        print("No layer found, this is a problem!")

    prod_pre = 1.0 # actually this is prod_all
    for i in range(m):
        lam = muns.ProtonMass/xs/lay[i, 3]
        prod_pre *= exp(-lay[i, 2] / lam)

    x0, x1, dx, rho = lay[m]
    lam = muns.ProtonMass/xs/rho
    termA = prod_pre/lam*exp(-(d-dmu-x0)/lam)*(1.-exp(-(d-dmu-x0)/ctau))
    
    for i in range(m+1):
        for j in range(i+2, m+1):
            dx  = lay[j-1,2]
            rho = lay[j-1,3]
            lam = muns.ProtonMass/xs/rho
            prods_mid[i,j] = prods_mid[i,j-1]*np.exp(-dx/ctau)

    # print(m)
    termB = 0.0
    for i in range(m):
        x0, x1, dx, rho = lay[i]
        lam = muns.ProtonMass/xs/rho
        termB += exp(-dx/lam)*(1.-exp(-dx/ctau))*prods_mid[i,m]/lam
    
    x0, x1, dx, rho = lay[m]
    lam = muns.ProtonMass/xs/rho
    termB *= prod_pre*exp(-(d-dmu-x0)*(1./ctau+1./lam))

    if (log(einimu/efin)/mub_water(einimu)/rhow > dw):
        factor = 1./(mub_rock(einimu)*rhor)
    else:
        factor = 1./(mub_water(einimu)*rhow)

    return (termB+termA)*factor/efin

cpdef double ProbMuon(double [:,:] lay, double efin, double einiN, double xs, double lt, bint is_T_absorbed):
    if is_T_absorbed:
        return ProbMuon_AbsorptionT(lay,efin,einiN,xs,lt)
    else:
        return ProbMuon_NoAbs(lay,efin,einiN,xs,lt)

