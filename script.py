#%%
import threebody

import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import imageio
from pathlib import Path
import os
from matplotlib.patches import FancyArrowPatch, Circle
import mpl_toolkits.mplot3d.art3d as art3d


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



t_upper = 3600*24*687

x_pnts1, y_pnts1,x_pnts2,y_pnts2,x_pnts3,y_pnts3,z_pnts1,z_pnts2,z_pnts3 = threebody.EDO.EDO_3.iter(t_upper)


t = 0


while t < len(x_pnts1):


    fig = plt.figure()
    ax = p3.Axes3D(fig)
    
    # ax.view_init(azim=(t*90)/593569)

    ax.get_proj = lambda: np.dot(p3.Axes3D.get_proj(ax), np.diag([1.6, 1.6, 0.9, 1]))


    plane = Circle((0,0,0), a2)
    Circle.set_color(plane,'0.9')
    Circle.set_alpha(plane, 0.1)
    ax.add_patch(plane)
    art3d.pathpatch_2d_to_3d(plane, z=0, zdir="z")



    ax.xaxis.set_pane_color((0.06, 0.06, 0.06, 0.99))
    ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 0.99))
    ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 0.99))



    ax.plot3D(x_pnts1, y_pnts1, z_pnts1, 'white', linewidth=1, alpha=1)
    ax.plot3D(x_pnts2, y_pnts2, z_pnts2, 'white', linewidth=1, alpha=1)

    ax.plot3D(x_pnts1, y_pnts1, z_pnts1, 'cornflowerblue', linewidth=0.9)
    ax.plot3D(x_pnts2, y_pnts2, z_pnts2, 'tomato', linewidth=0.9)
    

    #make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

   
    ax.set_xlim3d([-2.5e+11, 2.5e+11])
    ax.set_xlabel('X')

    ax.set_ylim3d([-2.5e+11, 2.5e+11])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1, 1])
    ax.set_zlabel('Z')

    
    ax.plot3D(0,0,0, 'o', markersize=40, color='darkorange')
    

    ax.plot(x_pnts1[t], y_pnts1[t], z_pnts1[t], 'o', markersize=7, color='cornflowerblue')  
    ax.plot(x_pnts2[t], y_pnts2[t], z_pnts2[t], 'o', markersize=5, color='tomato')  


    plt.pause(0.00000001) 

    t+=4000     


    #plt.savefig('EDO/Images/{}.png'.format(t))


    plt.close()




#image_path = Path('EDO/Images')


#images = list(image_path.glob('*.png'))
#image_list = []


# for file_name in images:

#     image_list.append(imageio.imread(file_name))


# imageio.mimwrite('EDO/Three_body.gif', image_list, fps=50)

# %%
