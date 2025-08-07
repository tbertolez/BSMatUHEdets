# Some useful functions 

import numpy as np
import MyUnits as muns
from scipy.interpolate import RectBivariateSpline

# CONVERSION FROM ELEVATION ANGLE TO EXIT POINT (ANITA)
# ------------------------------------------------------------

def get_anita_exit_angle_from_elv_angle(elev_angle,height):
    # elev_angle must be between -pi/2 and +pi/2
    # typical angles are around 6º-10º (negative, in radians)
    R = muns.EarthRadius*muns.RadioBend
    
    # if np.abs(np.cos(elev_angle)*(1+height/R)) <= 1:
        # we return an angle in radians, typically around 80º-90º
    return np.arcsin(np.cos(elev_angle)*(1+height/R))*muns.rads
    # else:
        # raise Exception("Angle above horizon, no exit point!")


def get_anita_exit_angle_error(elev_angle,elev_error,height):
    # elev_angle must be between -pi/2 and +pi/2
    # typical angles are around 6º-10º (negative, in radians)
    R = muns.EarthRadius*muns.RadioBend

    # we return the angular error in radians
    a  = elev_angle
    da = elev_error
    return -(1+height/R)*np.sin(a)/np.sqrt(1-(1+height/R)**2*np.cos(a)**2)*da


def get_anita_elv_angle_from_exit_angle(exit_angle,height):
    # exit_angle must be in radians, between 0 and pi
    # typical values are around 86º - 88º (in radians)
    R = muns.EarthRadius*muns.RadioBend

    # we return an angle in radians, typically around 6º-8º, negative
    return -np.arccos(np.sin(exit_angle)*R/(R+height))*muns.rads


def anita_convert_angle(angle, which_ang, height):
    """
    Converts from elevation angle to exit angle, or viceversa.
    
    Input:
    angle (float):  angle to convert. If elevation angle, should be negative.
    which_ang(str): 'elev' if angle is an elevation angle, 'exit' if angle is an exit angle.
    height (float): height at the time of detection.

    Output:
    a tuple of floats -> (elevation angle, exit angle)
    """
    if which_ang == 'elev':
        # angle is the elevation angle
        R = muns.EarthRadius*muns.RadioBend
        if np.abs(np.cos(angle)*(1+height/R)) > 1:
            raise Exception('Angle above horizon! Impossible to convert.')
        
        th = get_anita_exit_angle_from_elv_angle(angle, height)
        return angle, th
    
    elif which_ang == 'exit':
        # angle is the exit angle
        if angle > np.pi/2:
            raise Exception('Angle above horizon! Impossible to convert.')
  
        elv = get_anita_elv_angle_from_exit_angle(angle,height)
        return elv, angle

    else:
        # A wrong which_ang has been introduced
        print('A wrong angle type has been introduced.')
        raise Exception('Only elev or exit angles accepted.')  


# CONVERSION FROM ELEVATION ANGLE TO EXIT POINT (ICECUBE)
# ------------------------------------------------------------

def get_IC_exit_angle_from_nadir_angle(nad, d):
    # nad (the nadir angle, as seen from IC) must be between 0 and pi
    # 0 < nad < pi/2 is a northern-sky (up-going) neutrino, 
    # pi/2 < nad < pi is a southern-sky (down-going) neutrino.
    # typical angles are around 85º (in radians)

    R = muns.EarthRadius
    # d = pars.IC_Depth

    return np.arcsin((1-d/R)*np.sin(nad))

def get_IC_nadir_angle_from_exit_angle(exit, d, hemisphere = 'north'):
    # exit (the zenital angle, as seen from the exit point), is defined 0 < exit pi/2
    # this angle is degenerated: two different nadir angles (nad and 180-nad) produce
    # the same exit angle. We will need to think if this is any problem in the future.
    R = muns.EarthRadius
    # d = pars.IC_Depth

    if hemisphere == 'north':
        # the incoming neutrino came from the northern hemisphere
        # we return 0 < nadir < pi/2
        return np.arcsin((1-d/R)**-1 *np.sin(exit))
    if hemisphere == 'south':
        # the incoming neutrino came from the southern hemisphere
        # we return pi/2 < nadir < pi
        return np.pi - np.arcsin((1-d/R)**-1 *np.sin(exit))


def icecube_convert_angle(angle, which_ang, depth, hemisphere = 'north'):
    """
    Converts from nadir angle to exit angle, or viceversa.
    
    Input:
    angle (float):  angle to convert. In radians, must be positive.
    which_ang(str): 'nadir' if angle is seen from IC, 'exit' if angle is seen from exit point.

    Output:
    a tuple of floats -> (nadir angle, exit angle)
    """

    if which_ang == 'nadir':
        # angle is the nadir angle
        if angle < 0 or angle > np.pi:
            print('Please, use angles between 0 and pi.')
            raise Exception('Wrong nadir angle introduced.')

        ext = get_IC_exit_angle_from_nadir_angle(angle, depth)
        return angle, ext
    
    elif which_ang == 'exit':
        # angle is the exit angle
        R = muns.EarthRadius
        if np.sin(angle)/(1-depth/R) > 1:
            raise Exception('The introduced exit angle cannot be realized.')
  
        nad = get_IC_nadir_angle_from_exit_angle(angle, depth, hemisphere)
        return nad, angle

    else:
        print(which_ang)
        # A wrong which_ang has been introduced
        print('A wrong angle type has been introduced.')
        raise Exception('Only nadir or exit angles accepted.')  


# CONVERSION FROM ELEVATION ANGLE TO EXIT POINT (KM3NET)
# ------------------------------------------------------------

def get_k3_exit_angle_from_zenit_angle(zen,d):
    # zen (the zenit angle, as seen from km3net) must be between 0 and pi
    # 0 < zen < pi/2 is a northern-sky (down-going) neutrino, 
    # pi/2 < zen < pi is a southern-sky (up-going) neutrino.

    R = muns.EarthRadius
    # d = pars.K3_Depth

    return np.arcsin((1-d/R)*np.sin(zen))

def get_k3_zenit_angle_from_exit_angle(exit, d, hemisphere = 'north'):
    # exit (the zenital angle, as seen from the exit point), is defined 0 < exit < pi/2
    # this angle is degenerated: two different nadir angles (zen and 180-zen) produce
    # the same exit angle. We will need to think if this is any problem in the future.
    R = muns.EarthRadius
    # d = pars.IC_Depth

    if hemisphere == 'north':
        # the incoming neutrino came from the northern hemisphere
        # we return 0 < zenit < pi/2
        return np.arcsin((1-d/R)**-1 *np.sin(exit))
    
    if hemisphere == 'south':
        # the incoming neutrino came from the southern hemisphere
        # we return pi/2 < zenit < pi
        return np.pi - np.arcsin((1-d/R)**-1 *np.sin(exit))

def km3net_convert_angle(angle, which_ang, depth, hemisphere = 'south'):
    """
    Converts from nadir angle to exit angle, or viceversa.
    
    Input:
    angle (float):  angle to convert. In radians, must be positive.
    which_ang(str): 'zenit' if angle is seen from KM3NeT, 'exit' if angle is seen from exit point.
    hemisphere(str): 'north' for downgoing or 'south' for upgoing (where does the event come from)
    
    Output:
    a tuple of floats -> (zenit angle, exit angle)
    """

    if which_ang == 'zenit':
        # angle is the nadir angle
        if angle < 0 or angle > np.pi:
            print('Please, use angles between 0 and pi.')
            raise Exception('Wrong nadir angle introduced.')

        ext = get_k3_exit_angle_from_zenit_angle(angle, depth)
        return angle, ext
    
    elif which_ang == 'exit':
        # angle is the exit angle
        R = muns.EarthRadius
        if np.sin(angle)/(1-depth/R) > 1:
            raise Exception('The introduced exit angle cannot be realized.')
  
        nad = get_k3_zenit_angle_from_exit_angle(angle, depth, hemisphere)
        return nad, angle

    else:
        print(which_ang)
        # A wrong which_ang has been introduced
        print('A wrong angle type has been introduced.')
        raise Exception('Only nadir or exit angles accepted.')  



# EARTH-RELATED FUNCTIONS
# -----------------------------------

def ChordLength(theta):
    return 2*muns.EarthRadius*np.cos(theta)

def AnglesAtLayers():
    dat = muns.PREM_FullData()[:,0]
    angs = np.pi/2 - np.arcsin(dat/muns.EarthRadius)
    return angs

# print(90-AnglesAtLayers()/muns.degs)
# print(get_anita_elv_angle_from_exit_angle(np.pi/2.-AnglesAtLayers(), height = 38.5*muns.km)/muns.degs)

def LayersThroughChord(theta, end = None):
    """
    Computes the different layers that a particle will cross while travelling through
    an Earth's chord.

    Input:
    theta (float): exit angle, in radians, which defines the trajectory.
    end (float): the chord is not run completely, but stops at a distance
                 "end" before the exit point. Useful when the detector is underground (IC).

    Output:
    a numpy array where each row is a layer, and the columns are, for each layer: 
    initial position, final position, thickness and density.
    Layers are ordered from the furthest to the closest, 
    i.e. following the trajectory of the incoming particle.
    """
    # In the end, we're killing a fly with cannonballs with the "end" implementation. 
    # Since IC is in the PREM outer layer, there are only two possibilities:
    # either it is the last layer (up-going) or in the first one (down-going),
    # depending on the hemisphere of the angle. But whatever, this allows for
    # general Earth layers models, and arbitrary IC depth.

    # To improve: in terms of speed, if the direction is downgoing, there will just be one layer,
    # however I am computing all of them and removing all but one. Pretty inefficient...

    def r_to_xsup(r):
        return muns.EarthRadius*np.cos(theta) + np.sqrt(2)/2.*np.sqrt(2*r**2-muns.EarthRadius**2+muns.EarthRadius**2*np.cos(2*theta))

    def r_to_xinf(r):
        return muns.EarthRadius*np.cos(theta) - np.sqrt(2)/2.*np.sqrt(2*r**2-muns.EarthRadius**2+muns.EarthRadius**2*np.cos(2*theta))

    dat = muns.PREM_FullData()

    layers = []
    layers.append([0.0,dat[-1,1]]) # We add the first layer, with the last PREM density


    for i in range(dat.shape[0]-1):
        r = dat[i,0]
        if 2*r**2 > muns.EarthRadius**2-muns.EarthRadius**2*np.cos(2*theta):
            # print(r, r_to_xinf(r), r_to_xsup(r))
            layers.append([r_to_xinf(r),dat[i,1]])
            layers.append([r_to_xsup(r),dat[i,1]])

    # We add the last layer                
    layers.append([ChordLength(theta),dat[-1,1]])

    layers = np.array(layers)
    layers = layers[np.argsort(layers[:,0])] # we sort the data for increasing x

    lay = np.zeros((layers.shape[0]-1,4))
    lay[:,0] = layers[:-1,0] # Beginning of the layer
    lay[:,1] = layers[ 1:,0] # End of the layer
    lay[:,2] = np.diff(layers[:,0]) # Thickness of the layer

    ind = int(layers.shape[0]/2) # We want to remove one of the values of the central density, which is repeated
    lay[:,3] = np.concatenate((layers[:ind,1],layers[ind+1:,1])) # Density of the layer

    if end != None:
        i_end = np.argmax(lay[:,1] > ChordLength(theta)-end) + 1
        # i_end is the index of the layer which contains IC,
        # This is the last one the particle travels. 
        # The following should not be taken into account.
        lay = lay[:i_end]
        # The last layer is not run all over, but ends up before.
        lay[i_end-1,1] = ChordLength(theta)-end
        lay[i_end-1,2] = lay[i_end-1,1]-lay[i_end-1,0]
    


    return lay



# RELATED WITH THE DISTANCE TO THE EXIT POINT
# ------------------------------------------------

def DecayProbability(lt,length):
    return 1-np.exp(-length/(muns.c*lt))


def ANITA_DistanceToExitPoint(angle,height, which_ang = 'elev'):
    elv, th = anita_convert_angle(angle,which_ang,height)
    R = muns.EarthRadius*muns.RadioBend

    return -R*np.cos(th-elv)/np.cos(elv)

def ANITA_DistanceToExitPointSafe(th,elv):
    R = muns.EarthRadius*muns.RadioBend
    return -R*np.cos(th-elv)/np.cos(elv)


def IC_DistanceToExitPoint(nad,d):
    # nad (the nadir angle, as seen from IC) must be between 0 and pi
    # 0 < nad < pi/2 is a northern-sky (up-going) neutrino, 
    # pi/2 < nad < pi is a southern-sky (down-going) neutrino.
    # typical nad angles are around 85º (in radians)

    R = muns.EarthRadius

    exit = get_IC_exit_angle_from_nadir_angle(nad, d)

    a2 = 2*R**2*(1-np.cos(exit-nad))-2*R*d*(1-np.cos(exit-nad))+d**2
    return np.sqrt(a2)

def K3_DistanceToExitPoint(elev,d):
    # zen (the nadir angle, as seen from IC) must be between 0 and pi
    # 0 < zen < pi/2 is a northern-sky (down-going) neutrino, 
    # pi/2 < zen < pi is a southern-sky (up-going) neutrino.

    R = muns.EarthRadius
    tan = np.tan(elev)
    a02 = R**2 + (d-R)*((R-d)*np.cos(2*elev)+np.sin(2*elev)*np.sqrt(-d*(d-2*R)+R**2*tan**2))
    a12 = R**2 + (d-R)*((R-d)*np.cos(2*elev)-np.sin(2*elev)*np.sqrt(-d*(d-2*R)+R**2*tan**2))
    if elev > 0: # down-going muon
        return np.sqrt(np.max((a02,a12)))
    else: #up-going muon
        return np.sqrt(np.min((a02,a12)))

    a2 = 2*R**2*(1-np.cos(exit-zen))-2*R*d*(1-np.cos(exit-zen))+d**2
    return np.sqrt(a2)

# MISC FUNCTIONS
# --------------------------------

def is_floatable(element):
    """ Pretty straightforward:
            returns True if the argument is floatable,
            returns False otherwise.
    """
    try:
        float(element)
        return True
    except:
        return False


def get_data(fname, sep = ','):
    """
    Converts filename into array.
    """
    # inputfile = open(fname,'r+')
    # file_lines = inputfile.readlines()

    # mat = []
    # for line in file_lines:
    #     mat.append(line.strip().split(sep))
    # mat = np.array(mat).astype(np.float64)
    mat = np.loadtxt(fname, delimiter = sep)
    return mat

def get_sorted_data(fname):
    """ 
    This is used for reading probability tables, (xs,lt,th,probT,probN).
    Orders them in increasing th.
    """
    data = get_data(fname)
    data = data[np.argsort(data[:,2])] # we sort the data for increasing theta
    return data

def get_bestfit(data, index_to_minimize = -1):
    min_index = np.where(data[:,index_to_minimize] == np.min(data[:,index_to_minimize]))[0][0]
    bestfit = data[min_index]
    return bestfit

def get_difftomin(dat, index_to_minimize = -1):
    data = np.copy(dat)
    bestfit = get_bestfit(data,index_to_minimize)
    minvalue = bestfit[index_to_minimize]
    data[:,index_to_minimize] = data[:,index_to_minimize] - minvalue
    return data

def create_interpolator_from_datafile(fname, col_to_interp):
    # expect xs in the first column, lt in the second
    # col_to_interp admits one number if only one column, or a tuple for all the cols that want to be SUMMED
    data = np.nan_to_num(get_data(fname),nan=0.0)
    xs_data = np.sort(np.unique(data[:,0]))
    lt_data = np.sort(np.unique(data[:,1]))
    ordered_data = np.zeros((xs_data.shape[0],lt_data.shape[0]))

    if isinstance(col_to_interp,int):
        for row in data:
            x_index = np.where(xs_data == row[0])[0][0]
            y_index = np.where(lt_data == row[1])[0][0]
            ordered_data[x_index, y_index] = row[col_to_interp]
    else: # I assume that it's a list or a tuple
        for row in data:
            x_index = np.where(xs_data == row[0])[0][0]
            y_index = np.where(lt_data == row[1])[0][0]
            for col in col_to_interp:
                ordered_data[x_index, y_index] += row[col]
    
    return RectBivariateSpline(xs_data,lt_data,ordered_data)