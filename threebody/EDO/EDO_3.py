import math
import numpy as np
from scipy.integrate import odeint
from numba import jit
from os import path



@jit(nopython=True)
def distance(X, Y):

    """Calculate distance between two bodies

    Parameters
    ----------
    X: ndarray of shape (n, )
    Y: ndarray of shape (n, )

    Formula
    --------
    The formula used is the euclidian distance:
        formula: d(X,Y) = sqrt(sum(x_i - y_i)), wher i goes from 1 to len(X)
    """

    return math.sqrt(np.sum((X-Y)**2))


def velocities(G=6.67e-11, M, r):
    """Return the velocity of a body

    Parameters
    -----------

    G: scalar, default = 6.67e-11
        gravitanionnal constant

    M: scalar,
        mass of body

    r: scalar, 
        distance from the body to another that attracts it
    """

    return math.sqrt(G*M/r)


@jit(nopython=True)
def f(r, t, G=6.67e-11, m1=5.972e+24, m2=6.4185e+23, m3=1.989e+30,
        AU=1.496e+11, a1=1.0*1.496e+11, a2=1.52*1.496e+11,
        x_i1=-1.0*1.496e+11, y_i1=0, z_i1 = 0,
        v_x1i=0, v_y1i=29779.301841746023, v_z1i=0,
        x_i2=1.52*1.496e+11, y_i2=0, z_i2=0, v_x2i=0,
        v_y2i=24154.203325249873, v_z2i=0, 
        x_i3=0, y_i3=0, z_i3=0, v_x3i=0, v_y3i=0, v_z3i=0):

    """Return the derivative of differential equation system for 3 body-problem

    Parameters
    ----------

    r: ndarray of shape (n, ),
        vector which positions, velocities and accelerations of three bodies

    t: scalar,
        which represent time

    G: scalar, default = 6.67e-11 (N.M^2.kg^(-2))
        represent gravitational constant

    m1: scalar, default = 5.972e+24 (Earth mass in kg)
        Mass of first body

    m2: Mass of second body
        default: m2 = 6.4185e+23 (Mars mass in kg)

    m3: Mass of third body
        default: m3 = 1.989e+30 (Sun mass in kg)

    AU: scalar, default = 1.496e+11
        Astronomical unit

    x_i1: scalar, default = -1.0*1.496e+11
        The initial position in x direction of the first body

    y_i1: scalar, default = 0
        The initial position in y direction of the first body

    z_i1: scalar, default = 0
        The initial position in z direction of the first body

    v_x1i: scalar, default = 0
        The initial velocity in x direction of the first body

    v_y1i: scalar,  default = 29779.301841746023
        The initial velocity in y direction of the first body

    v_z1i: scalar, default = 0
        The initial velocity in z direction of the first body

    x_i2: scalar, default = 1.52*1.496e+11
        The initial position in x direction of the second body

    y_i2: scalar, default = 0
        The initial position in y direction of the second body

    z_i2: scalar, default = 0
        The initial position in z direction of the second body

    v_x2i: scalar, default = 0
        The initial velocity in x direction of the second body

    v_y2i: scalar,  default = 24154.203325249873
        The initial velocity in y direction of the second body

    v_z2i: scalar, default = 0
        The initial velocity in z direction of the second body

    x_i3: scalar, default = 0
        The initial position in x direction of the third body

    y_i3: scalar, default = 0
        The initial position in y direction of the third body

    z_i3: scalar, default = 0
        The initial position in z direction of the third body

    v_x3i: scalar, default = 0
        The initial velocity in x direction of the third body

    v_y3i: scalar,  default = 0
        The initial velocity in y direction of the third body

    v_z3i: scalar, default = 0
        The initial velocity in z direction of the third body


    Problem solved
    --------------
    The systeme of equation solve is as follows:
    We note ri = (xi, yi, zi) the position of the body i.

    (d^2)r1 = -Gm2*(r1-r2)/(distance(r1-r2))**3 -Gm2*(r1-r3)/(distance(r1-r3))**3
    (d^2)r2 = -Gm2*(r2-r3)/(distance(r2-r3))**3 - Gm2*(r2-r1)/(distance(r2-r1))**3
    (d^2)r3 = -Gm2*(r3-r1)/(distance(r3-r1))**3 - Gm2*(r3-r2)/(distance(r3-r2))**3

    Where (d^2)ri means the second derivative of r 
    of body i (with respect to time), G the gravitational constant,
    mi the mass of body i.

    The equation system above, composed of 9 equations of order 2, 
    is transformed into an equation system of 9 differential equations 
    of order 1.

    Note 1
    ----
    The velocity v of a body with a mass M is given by: v = sqrt(G*M/r),
    where r it's a distance from a massive body (typically a star).

    Note 2
    -------
    If we consider the Sun and Earth, the orbital volocity of Earth is given
    by the sum of the Earth's velocity without Sun's gravitational influence 
    and velocity which resulted from Sun's gravity.
    It's the same thing for Mars.
    """


    x1 = r[0]
    y1 = r[1]
    v_x1 = r[2]
    v_y1 = r[3]

    x2 = r[4]
    y2 = r[5]
    v_x2 = r[6]
    v_y2 = r[7]

    x3 = r[8]
    y3 = r[9]
    v_x3 = r[10]
    v_y3 = r[11]

    z1 = r[12]
    z2 = r[13]
    z3 = r[14]

    v_z1 = r[15]
    v_z2 = r[16]
    v_z3 = r[17]


    r1 = np.array([x1, y1, z1])
    r2 = np.array([x2, y2, z2])
    r3 = np.array([x3, y3, z3])

    dr1 = v_x1
    dr2 = v_y1

    dr3 = (G*m2/distance(r1, r2)**3)*(x2-x1) + (G*m3/distance(r1, r3)**3)*(x3-x1)
    dr4 = (G*m2/distance(r1, r2)**3)*(y2-y1) + (G*m3/distance(r1, r3)**3)*(y3-y1)

    dr5 = v_x2
    dr6 = v_y2

    dr7 = (G*m1/distance(r1, r2)**3)*(x1-x2) + (G*m3/distance(r2, r3)**3)*(x3-x2)
    dr8 = (G*m1/distance(r1, r2)**3)*(y1-y2) + (G*m3/distance(r2, r3)**3)*(y3-y2)

    dr9 = v_x3
    dr10 = v_y3

    dr11 = (G*m1/distance(r1, r3)**3)*(x1-x3) + (G*m2/distance(r2, r3)**3)*(x2-x3)
    dr12 = (G*m1/distance(r1, r3)**3)*(y1-y3) + (G*m2/distance(r2, r3)**3)*(y2-y3)

    dr13 = v_z1
    dr14 = v_z2
    dr15 = v_z3

    dr16 = (G*m2/distance(r1, r2)**3)*(z2-z2) + (G*m3/distance(r1, r3)**3)*(z3-z1)
    dr17 = (G*m3/distance(r2, r3)**3)*(z1-z2) + (G*m1/distance(r2, r1)**3)*(z1-z2)
    dr18 = (G*m1/distance(r1, r3)**3)*(z1-z3) + (G*m2/distance(r2, r3)**3)*(z2-z3)


    dr = np.array([dr1, dr2, dr3, dr4, dr5, dr6,
                dr7, dr8, dr9, dr10, dr11, dr12,
                dr13, dr14, dr15, dr16, dr17, dr18])


    return dr



@jit(nopython=True)
def trajectories(t_upper=3600*24*687, h=100):

    """Return the coordinates of the trajectories of three bodies

    Parameters
    ----------
    t_upper: scalar, default = 3600*24*687
        The upper bound of time.


    h: scalar,  default = 100  (Not recomended to change, it can make the algorithm instable)
        The step size for RK4 algorithm.


    Note
    ----------
    t_upper is put at 24*3600*687 to simulate a marsian year.
    It can be put at 24*3600*365 to simulate a earth year. 

    Method used
    ----------
    This function uses the RK4 (Runge Kutta 4) method to solve the differential system 
    composed of 18 equations of order 1 of the 3 body problem.
    The RK4 method allows to solve the differential equation y' = f(y,t),
    where y and t can be a scalar or vector and y' is the derivative of y.
    In physic's problem, t represent generally the time, this is the case 
    for three body problem
    """

    x_i1 = 1.0*1.496e+11     # initial values for planet 1 in x, y and z direction
    y_i1 = 0
    v_x1i = 0
    v_y1i = 29779.301841746023
    z_i1 = 0
    v_z1i = 0

    x_i2 = 1.52*1.496e+11     # initial values for planet 2 in x, y and z direction
    y_i2 = 0
    v_x2i = 0
    v_y2i = 24154.203325249873
    z_i2 = 0
    v_z2i = 0


    x_i3 = 0       # initial values for Sun in x, y and z direction
    y_i3 = 0
    v_x3i = 0
    v_y3i = 0
    z_i3 = 0
    v_z3i = 0


    r = np.array([x_i1, y_i1, v_x1i, v_y1i, x_i2,
    y_i2, v_x2i, v_y2i, x_i3, y_i3, v_x3i, v_y3i, 
    z_i1, z_i2, z_i3, v_z1i, v_z2i, v_z3i])     # Initial positions and velocities


    # We create vectors which will contains the trajectories
    # and velocities of each bodies
    x_pnts1 = [x_i1]
    y_pnts1 = [y_i1]
    v_x_pnts1 = [v_x1i]
    v_y_pnts1 = [v_y1i]

    x_pnts2 = [x_i2]
    y_pnts2 = [y_i2]
    v_x_pnts2 = [v_x2i]
    v_y_pnts2 = [v_y2i]

    x_pnts3 = [x_i3]
    y_pnts3 = [y_i3]
    v_x_pnts3 = [v_x3i]
    v_y_pnts3 = [v_y3i]

    x_pnts3 = [x_i3]
    y_pnts3 = [y_i3]
    v_x_pnts3 = [v_x3i]
    v_y_pnts3 = [v_y3i]

    z_pnts1 = [z_i1]
    z_pnts2 = [z_i2]
    z_pnts3 = [z_i3]

    v_z_pnts1 = [v_z1i]
    v_z_pnts2 = [v_z2i]
    v_z_pnts3 = [v_z3i]


    # We create a vector which will contain the time
    t_i = 0.0      # Initial value
    t_values = [t_i]


    for t in range(0, t_upper, h):

        # We used the RK4 formula here
        k1 = h*f(r=r, t=0)
        k2 = h*f(r=r + 0.5*k1, t=t + (h/2))
        k3 = h*f(r=r + 0.5*k2, t=t + (h/2))
        k4 = h*f(r=r + h*k3, t=t+h)

        # We calculate the new vector r
        r += (k1 + 2*k2 + 2*k3 + k4)*(1.0/6.0)

        # We add the new points calculated
        x_pnts1.append(r[0])
        y_pnts1.append(r[1])

        v_x_pnts1.append(r[2])
        v_y_pnts1.append(r[3])

        x_pnts2.append(r[4])
        y_pnts2.append(r[5])
        v_x_pnts2.append(r[6])
        v_y_pnts2.append(r[7])


        x_pnts3.append(r[8])
        y_pnts3.append(r[9])
        v_x_pnts3.append(r[10])
        v_y_pnts3.append(r[11])


        z_pnts1.append(r[12]) 
        z_pnts2.append(r[13]) 
        z_pnts3.append(r[14]) 

        v_z_pnts1.append(r[15]) 
        v_z_pnts2.append(r[16]) 
        v_z_pnts3.append(r[17])


        t_values.append(t)


    # We return all the trajectories
    return x_pnts1, y_pnts1, x_pnts2, y_pnts2, x_pnts3, y_pnts3, z_pnts1, z_pnts2, z_pnts3


# x_pnts1, y_pnts1, x_pnts2, y_pnts2, x_pnts3, y_pnts3, z_pnts1, z_pnts2, z_pnts3 = trajectories()
