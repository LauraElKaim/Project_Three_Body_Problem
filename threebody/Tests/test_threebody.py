import pytest
import numpy as np
from threebody.EDO.EDO_3 import distance
from threebody.EDO.EDO_3 import trajectories
from threebody.EDO.EDO_3 import velocity
from threebody.EDO.EDO_3 import derivative
pytest.__path__



def test_distance():

    """We test function distance from EDO_3 file"""

    X = np.array([0, 1])
    Y = np.array([0, 2])

    assert (distance(X,Y) == 1)


def test_trajectories():

    """We test function trajectories from EDO_3 file"""

    x_pnts1, y_pnts1, x_pnts2, y_pnts2, x_pnts3, y_pnts3, z_pnts1, z_pnts2, z_pnts3 = trajectories(h=50000)
    size = len(x_pnts1)

    assert (len(x_pnts2) == len(x_pnts3) == size)
    assert (len(y_pnts1) == len(y_pnts2) == len(y_pnts3) == size)
    assert (len(z_pnts1) == len(z_pnts2) == len(z_pnts3) == size)


def test_velocity():

    """We test function velocity from EDO_3 file"""

    assert (velocity(M=1.989e+30, r=1.0*1.496e+11) == 29779.301841746023)


def test_derivative():

    """We test function derivative from EDO_3 file"""

    r_test = np.arange(18)

    dr = derivative(r=r_test, t=0)

    list_test = [2.00000000e+00, 3.00000000e+00, 6.99826313e+17,
                    6.99826313e+17, 6.00000000e+00, 7.00000000e+00, 2.79929323e+18,
                    2.79929323e+18, 1.00000000e+01, 1.10000000e+01, -3.00435858e+12,
                    -3.00435858e+12, 1.50000000e+01, 1.60000000e+01, 1.70000000e+01,
                    1.74956352e+17, -6.99827511e+17, -7.51089644e+11]

    assert (np.allclose(list_test, dr, rtol=1.e-8, atol=1.e-8))
