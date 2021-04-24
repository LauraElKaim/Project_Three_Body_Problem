import math
import numpy as np
from numba import jit


@jit(nopython=True)
def distance(X, Y):

    """Calculate distance between two bodies

    :param X: coordinate of body one
    :type X: ndarray of shape (n, 1)
    :param Y: coordinate of body two
    :type Y: ndarray of shape (n, 1)

    :return: Euclidian distance between body one and two
    :rtype: float

    """

    return math.sqrt(np.sum((X-Y)**2))


@jit(nopython=True)
def velocity(M, r, G=6.67e-11):

    """
    Return the velocity of a body

    :param G: Gravitanionnal constant, default = 6.67e-11
    :rtype: float

    :param M: Mass of body that attracts it (in `kg`)
    :type M: float

    :param r: Distance from the body to another that attracts it
    :type r: float

    :return: Velocity of body
    :rtype: float
    """

    if M < 0:
        return print(f"The given mass is negative")

    if r < 0:
        print(f"The given distance is negative")

    if G < 0:
        print(f"The gravitational constant is negative")

    return np.sqrt(G*M/r)


@jit(nopython=True)
def derivative(r, t, G=6.67e-11, AU=1.496e+11,
               m1=5.972e+24, m2=6.417e+23, m3=1.989e+30,
               a1=1.0*1.496e+11, a2=1.52*1.496e+11):

    """
    Return the derivative of differential equation system for 3 body-problem

    :param r: Vector which contain positions, \n
              velocities and accelerations of three bodies
    :type r: ndarray of shape (n, 1)

    :param t: Time
    :type t: float

    :param G: Represent gravitational constant, \n
              default = 6.67e-11 (N.M^2.kg^(-2))
    :type G: float

    :param AU: Astronomical unit, default = 1.496e+11
    :type AU: float

    :param m1: Mass of first body, default = 5.972e+24 (Earth mass in kg)
    :type m1: float

    :param m2: Mass of second body, default = 6.4185e+23  (Mars mass in kg)
    :type m2: float

    :param m3: Mass of third body, default =  1.989e+30 (Sun mass in kg)
    :type m3: float

    :param a1: distance of body 1 from the body center, default = 1.0*1.496e+11
    :type a1: float

    :param a2: distance of body 2 from the body center, \n
               default = 1.52*1.496e+11
    :type a1: float

    :return: The right hand side of system of differential equation \n
            of thre body problem
    :rtype: Vectors
    """

    if G < 0:
        print(f"The gravitational constant is negative")

    if AU < 0:
        print(f"The Astronomical unit is negative")

    if m1 < 0:
        print(f"The mass of the first body is negative")

    if m2 < 0:
        print(f"The mass of the second body is negative")

    if m3 < 0:
        print(f"The mass of the third body is negative")

    if a1 < 0:
        print(f"The distance of body 1 from the body center is negative")

    if a2 < 0:
        print(f"The distance of body 2 from the body center is negative")

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

    dr3 = (G*m2/distance(r1, r2)**3)*(x2-x1)
           + (G*m3/distance(r1, r3)**3)*(x3-x1)
    dr4 = (G*m2/distance(r1, r2)**3)*(y2-y1)
           + (G*m3/distance(r1, r3)**3)*(y3-y1)

    dr5 = v_x2
    dr6 = v_y2

    dr7 = (G*m1/distance(r1, r2)**3)*(x1-x2)
           + (G*m3/distance(r2, r3)**3)*(x3-x2)
    dr8 = (G*m1/distance(r1, r2)**3)*(y1-y2)
           + (G*m3/distance(r2, r3)**3)*(y3-y2)

    dr9 = v_x3
    dr10 = v_y3

    dr11 = (G*m1/distance(r1, r3)**3)*(x1-x3)
            + (G*m2/distance(r2, r3)**3)*(x2-x3)
    dr12 = (G*m1/distance(r1, r3)**3)*(y1-y3)
            + (G*m2/distance(r2, r3)**3)*(y2-y3)

    dr13 = v_z1
    dr14 = v_z2
    dr15 = v_z3

    dr16 = (G*m2/distance(r1, r2)**3)*(z2-z2)
            + (G*m3/distance(r1, r3)**3)*(z3-z1)
    dr17 = (G*m3/distance(r2, r3)**3)*(z1-z2)
            + (G*m1/distance(r2, r1)**3)*(z1-z2)
    dr18 = (G*m1/distance(r1, r3)**3)*(z1-z3)
            + (G*m2/distance(r2, r3)**3)*(z2-z3)

    dr = np.array([dr1, dr2, dr3, dr4, dr5, dr6,
                   dr7, dr8, dr9, dr10, dr11, dr12,
                   dr13, dr14, dr15, dr16, dr17, dr18])

    return dr


@jit(nopython=True)
def trajectories(t_upper=3600*24*687, h=100, m1=5.972e+24, m2=6.417e+23,
                 m3=1.989e+30, a1=1.0*1.496e+11, a2=1.52*1.496e+11):

    """
    Return the coordinates of the trajectories of three bodies

    :param t_upper: The upper bound of time, default = 3600*24*687
    :type t_upper: float

    :param h: The step size for RK4 algorithm, default = 100
    :type h: float   The step size for RK4 algorithm

    :param m1: Mass of first body, default = 5.972e+24 (Earth mass in kg)
    :type m1: float

    :param m2: Mass of second body, default = 6.4185e+23  (Mars mass in kg)
    :type m2: float

    :param m3: Mass of third body, default = 1.989e+30 (Sun mass in kg)
    :type m3: float

    :param a1: distance of body 1 from the body center
    :type a1: float

    :param a2: distance of body 2 from the body center
    :type a1: float

    :return: All the trajectories of three bodies
    :rtype: Vectors
    """

    # We check if parameters are all positive

    list_parameters = [t_upper, h, m1, m2, m3,
                       a1, a2]

    for parameters in list_parameters:

        if parameters < 0:
            print(f'You have entered a negative parameter')

    #   initial values for planet 1 in x, y and z direction
    x_i1 = a1
    y_i1 = 0
    v_x1i = 0
    v_y1i = 29779.301841746023
    z_i1 = 0
    v_z1i = 0

    # initial values for planet 2 in x, y and z direction
    x_i2 = a2
    y_i2 = 0
    v_x2i = 0
    v_y2i = 24154.203325249873
    z_i2 = 0
    v_z2i = 0

    # initial values for Sun in x, y and z direction
    x_i3 = 0
    y_i3 = 0
    v_x3i = 0
    v_y3i = 0
    z_i3 = 0
    v_z3i = 0

# Initial positions and velocities
    r = np.array([x_i1, y_i1, v_x1i, v_y1i, x_i2,
                  y_i2, v_x2i, v_y2i, x_i3, y_i3, v_x3i, v_y3i,
                  z_i1, z_i2, z_i3, v_z1i, v_z2i, v_z3i])

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

    m1 = m1
    m2 = m2
    m3 = m3
    a1 = a1
    a2 = a2

    # We create a vector which will contain the time
    # Initial value
    t_i = 0.0
    t_values = [t_i]

    for t in range(0, t_upper, h):

        # We used the RK4 formula here
        k1 = h*derivative(r=r, t=0,  m1=5.972e+24, m2=m2, m3=1.989e+30,
                          a1=a1, a2=1.52*1.496e+11)
        k2 = h*derivative(r=r + 0.5*k1, t=t + (h/2), m1=5.972e+24,
                          m2=6.417e+23, m3=1.989e+30, a1=1.0*1.496e+11,
                          a2=1.52*1.496e+11)
        k3 = h*derivative(r=r + 0.5*k2, t=t + (h/2), m1=5.972e+24
                          m2=6.417e+23, m3=1.989e+30, a1=1.0*1.496e+11,
                          a2=1.52*1.496e+11)
        k4 = h*derivative(r=r + h*k3, t=t+h, m1=5.972e+24, m2=6.417e+23,
                          m3=1.989e+30, a1=1.0*1.496e+11, a2=1.52*1.496e+11)

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

# x_pnts1, y_pnts1, x_pnts2, y_pnts2, x_pnts3, y_pnts3, z_pnts1, z_pnts2, z_pnts3 = trajectories(h=100000)
