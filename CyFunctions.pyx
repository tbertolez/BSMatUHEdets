import MyUnits as muns
cimport numpy as np
import numpy as np
from libc.math cimport sqrt, cos, exp, pi

cpdef double exp_weight(double x, double mu, double sig):
    # An expontential weight for averaging around observed events
    return exp(-(x-mu)**2/2/sig**2)/sqrt(2*pi)/sig

cpdef double ChordLength(double theta):
    # Distance travelled by a particle which exits the Earth at an angle theta
    return 2*muns.EarthRadius*cos(theta)

cdef double r_to_xsup(double r, double theta):
    # First crossing of the trajectory with spherical layers
    return muns.EarthRadius * cos(theta) + sqrt(2) / 2.0 * sqrt(2 * r ** 2 - muns.EarthRadius ** 2 + muns.EarthRadius ** 2 * cos(2 * theta))

cdef double r_to_xinf(double r, double theta):
    # Second crossing of the trajectory with spherical layers
    return muns.EarthRadius * cos(theta) - sqrt(2) / 2.0 * sqrt(2 * r ** 2 - muns.EarthRadius ** 2 + muns.EarthRadius ** 2 * cos(2 * theta))

cpdef np.ndarray LayersThroughChord(double theta, double end=0.0):
    # Returns the layers through which the particle crosses.
    # end is useful for underground detectors, where not the full chord length is travelled
    cdef np.ndarray dat, layers2
    cdef list layers
    cdef double r
    cdef int i, ind, i_end
    cdef np.ndarray lay

    dat = muns.PREM_FullData()
    layers = []
    layers.append([0.0, dat[-1, 1]])

    for i in range(dat.shape[0] - 1):
        r = dat[i, 0]
        if 2 * r ** 2 > muns.EarthRadius ** 2 - muns.EarthRadius ** 2 * cos(2 * theta):
            layers.append([r_to_xinf(r, theta), dat[i, 1]])
            layers.append([r_to_xsup(r, theta), dat[i, 1]])

    layers.append([ChordLength(theta), dat[-1, 1]])

    layers2 = np.array(layers)
    layers2 = layers2[np.argsort(layers2[:, 0])]

    lay = np.zeros((layers2.shape[0] - 1, 4))
    lay[:, 0] = layers2[:-1, 0]
    lay[:, 1] = layers2[1:, 0]
    lay[:, 2] = np.diff(layers2[:, 0])

    ind = int(layers2.shape[0] / 2)
    lay[:, 3] = np.concatenate((layers2[:ind, 1], layers2[ind + 1:, 1]))

    if end > 0.0:
        i_end = np.argmax(lay[:, 1] > ChordLength(theta) - end) + 1
        lay = lay[:i_end]
        lay[i_end - 1, 1] = ChordLength(theta) - end
        lay[i_end - 1, 2] = lay[i_end - 1, 1] - lay[i_end - 1, 0]

    return lay
