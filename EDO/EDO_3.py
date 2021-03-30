import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numba import jit
from celluloid import Camera
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from PIL import Image



G = 6.67e-11               # Gravitational constant
m3 = 1.989e+30             # mass of the sun
m1 = 5.972e+24             # mass of planet 1 (Earth)
m2 = 6.4185e+23            # mass of planet 2 (Mars)
AU = 1.496e+11             # Astronomical unit
a1 = 1.0*AU         # Distance from planet 1 to the sun
a2 = 1.52*AU        # Distance from planet 2 to the sun


x_i1 = a1       # initial values for planet 1 in x, y and z direction
y_i1 = 0
v_x1i = 0
v_y1i = 29779.301841746023        
z_i1 = 0
v_z1i = 0

x_i2 = a2      # initial values for planet 2 in x, y and z direction
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

t = 0
t_i = 0.0
t_upper = 3600*24*687     # A Martian year is 687 Earth days                  


r = np.array([x_i1, y_i1, v_x1i, v_y1i, x_i2,
y_i2, v_x2i, v_y2i, x_i3, y_i3, v_x3i, v_y3i, 
z_i1,z_i2,z_i3,v_z1i,v_z2i,v_z3i])     # Initial positions and velocities



@jit(nopython=True)
def distance(X, Y):
    """Calculate distance between two bodies"""
    return math.sqrt(np.sum((X-Y)**2))



@jit(nopython=True)
def f(r, t):
    """Return the derivative of differential equation system"""

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
    
    dr3 = (G*m2/distance(r1,r2)**3)*(x2-x1) + (G*m3/distance(r1,r3)**3)*(x3-x1)
    dr4 = (G*m2/distance(r1,r2)**3)*(y2-y1) + (G*m3/distance(r1,r3)**3)*(y3-y1)

    dr5 = v_x2
    dr6 = v_y2

    dr7 = (G*m1/distance(r1,r2)**3)*(x1-x2) + (G*m3/distance(r2,r3)**3)*(x3-x2)
    dr8 = (G*m1/distance(r1,r2)**3)*(y1-y2) + (G*m3/distance(r2,r3)**3)*(y3-y2)

    dr9 = v_x3
    dr10 = v_y3

    dr11 = (G*m1/distance(r1,r3)**3)*(x1-x3) + (G*m2/distance(r2,r3)**3)*(x2-x3)
    dr12 = (G*m1/distance(r1,r3)**3)*(y1-y3) + (G*m2/distance(r2,r3)**3)*(y2-y3)

    dr13 = v_z1
    dr14 = v_z2
    dr15 = v_z3

    dr16 = (G*m2/distance(r1,r2)**3)*(z2-z2) + (G*m3/distance(r1,r3)**3)*(z3-z1)
    dr17 = (G*m3/distance(r2,r3)**3)*(z1-z2) + (G*m1/distance(r2,r1)**3)*(z1-z2)
    dr18 = (G*m1/distance(r1,r3)**3)*(z1-z3) + (G*m2/distance(r2,r3)**3)*(z2-z3)

    dr = np.array([dr1, dr2, dr3, dr4, dr5, dr6, dr7, dr8, dr9, dr10, dr11, dr12,
                   dr13, dr14, dr15, dr16, dr17, dr18])


    return dr


h = 100
print('Pas =',h)


@jit(nopython=True)
def iter(t_upper):
    """Return the coordinates of the trajectories of three bodies"""

    r = np.array([x_i1, y_i1, v_x1i, v_y1i, x_i2,
    y_i2, v_x2i, v_y2i, x_i3, y_i3, v_x3i, v_y3i, 
    z_i1,z_i2,z_i3,v_z1i,v_z2i,v_z3i])


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


    t_values = [t_i]


    for t in range(0,t_upper,h):


        k1 = h*f(r, t)                
        k2 = h*f(r + 0.5*k1, t + (h/2))
        k3 = h*f(r + 0.5*k2, t + (h/2))
        k4 = h*f(r + h*k3, t+h)

        r += (k1 + 2*k2 + 2*k3 + k4)*(1.0/6.0)


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


    return x_pnts1,y_pnts1,x_pnts2,y_pnts2,x_pnts3,y_pnts3,z_pnts1,z_pnts2,z_pnts3

  
# x_pnts1, y_pnts1,x_pnts2,y_pnts2,x_pnts3,y_pnts3,z_pnts1,z_pnts2,z_pnts3 = iter(t_upper)
