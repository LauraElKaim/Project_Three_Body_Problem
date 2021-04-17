import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numba import jit
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from PIL import Image
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import imageio
from pathlib import Path
import os
import os.path
from matplotlib.patches import FancyArrowPatch, Circle
import mpl_toolkits.mplot3d.art3d as art3d
from os import path
from random import randint




# We test if the import of a package was successful or not
# If not, we raise an exception

try:

    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    from numba import jit
    from mpl_toolkits import mplot3d
    import mpl_toolkits.mplot3d.axes3d as p3
    from PIL import Image
    import mpl_toolkits.mplot3d.axes3d as p3
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import numpy as np
    import imageio
    from pathlib import Path
    import os
    import os.path
    from matplotlib.patches import FancyArrowPatch, Circle
    import mpl_toolkits.mplot3d.art3d as art3d
    from os import path
    from random import randint


except Exception as error:
    print('Caught an exception', error)





@jit(nopython=True)
def distance(X, Y):

    """Calculate distance between two bodies
    
    Parameters
    ----------

    X : ndarray of shape (n, )  
    Y : ndarray of shape (n, )

    Formula
    --------

    The formula used is the euclidian distance:
        formula: d(X,Y) = sqrt(sum(x_i - y_i)), wher i goes from 1 to len(X)
    
    
    """

    return math.sqrt(np.sum((X-Y)**2))



@jit(nopython=True)
def f(r, t, G=6.67e-11, m1 = 5.972e+24, m2=6.4185e+23, m3=1.989e+30, 
        AU=1.496e+11, a1 = 1.0*1.496e+11, a2=1.52*1.496e+11,
        x_i1=-1.0*1.496e+11, y_i1=0, z_i1 = 0,
        v_x1i=0, v_y1i= 29779.301841746023, v_z1i=0,
        x_i2=1.52*1.496e+11, y_i2=0, z_i2=0, v_x2i=0,
        v_y2i=24154.203325249873, v_z2i=0, 
        x_i3=0, y_i3=0, z_i3=0, v_x3i=0, v_y3i=0, v_z3i=0):
        
    """Return the derivative of differential equation system for 3 body-problem

    Parameters
    ----------
    
    r : ndarray of shape (n,), vector which positions, velocities and accelerations of three bodies

    t : scalar which represent time

    G: scalar, represent gravitational constant
        default: G = 6.67e-11 (N.M^2.kg^(-2))

    m1: Mass of first body
        default: m1 = 5.972e+24 (Earth mass in kg)

    m2: Mass of second body
        default: m2 = 6.4185e+23 (Mars mass in kg)
    
    m3: Mass of third body  
        default: m3 = 1.989e+30 (Sun mass in kg)

    AU: Astronomical unit 
        default: AU = 1.496e+11

    x_i1: The initial position in x direction of the first body  
        default: x_i1 = 0
    
    y_i1: The initial position in y direction of the first body    
        default: y_i1 = 0
    
    z_i1: The initial position in z direction of the first body    
        default: z_i1 = 0
    
    v_x1i : The initial velocity in x direction of the first body    
        default: v_x1i = 0
    
    v_y1i : The initial velocity in y direction of the first body    
        default: v_y1i = 29779.301841746023

    v_z1i : The initial velocity in z direction of the first body    
        default: v_z1i = 0
   
    x_i1: The initial position in x direction of the first body  
        default: x_i1 = 0
    
    y_i1: The initial position in y direction of the first body    
        default: y_i1 = 0
    
    z_i1: The initial position in z direction of the first body    
        default: z_i1 = 0
 
    v_x1i : The initial velocity in x direction of the first body    
        default: v_x1i = 0
    
    v_y1i : The initial velocity in y direction of the first body    
        default: v_y1i = 29779.301841746023

    v_z1i : The initial velocity in z direction of the first body    
        default: v_z1i = 0

    x_i2: The initial position in x direction of the second body  
        default: x_i2 = 0
    
    y_i2: The initial position in y direction of the second body    
        default: y_i2 = 1.52*1.496e+11
    
    z_i2: The initial position in z direction of the second body    
        default: z_i2 = 0
    
    v_x2i : The initial velocity in x direction of the second body    
        default: v_x2i = 0
    
    v_y2i : The initial velocity in y direction of the second body    
        default: v_y2i = 24154.203325249873

    v_z2i : The initial velocity in z direction of the second body    
        default: v_z2i = 0
    
    
    x_i3: The initial position in x direction of the third body  
        default: x_i3 = 0
    
    y_i3: The initial position in y direction of the third body    
        default: y_i3 = 1.52*1.496e+11
    
    z_i3: The initial position in z direction of the third body    
        default: z_i3=0
    
    v_x3i : The initial velocity in x direction of the third body    
        default: v_x3i = 0
    
    v_y3i : The initial velocity in y direction of the third body    
        default: v_y3i = 0

    v_z3i : The initial velocity in z direction of the third body    
        default: v_z3i = 0
    

    Problem solved
    --------------

    The systeme of equation solve is as follows:

    We note ri = (xi, yi, zi) the position of the body i.
    
    (d^2)r1 = -Gm2*(r1-r2)/(distance(r1-r2))**3 - -Gm2*(r1-r3)/(distance(r1-r3))**3
    (d^2)r2 = -Gm2*(r2-r3)/(distance(r2-r3))**3 - -Gm2*(r2-r1)/(distance(r2-r1))**3
    (d^2)r3 = -Gm2*(r3-r1)/(distance(r3-r1))**3 - -Gm2*(r3-r2)/(distance(r3-r2))**3

    Where (d^2)ri means the second derivative of r of body i (with respect to time),
    G the gravitational constant, mi the mass of body i.

    The equation system above, composed of 9 equations of order 2, is transformed
    into an equation system of 9 differential equations of order 1.


    Note 1
    ----

    The velocity v of a body with a mass M is given by: v = sqrt(G*M/r),
    where r it's a distance from a massive body (typically a star).


    Note 2
    -------

    If we consider the Sun and Earth, the orbital volocity of Earth is given
    by the sum of the Earth's velocity without Sun's gravitational influence and velocity which
    resulted from Sun's gravity.
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


# Appelé RK4 trajectories avec tout les parametres
# Creer des fonctions pour chaque planetes p1, p2, p3

# We use numba to accelerate the numerical calcul

@jit(nopython=True)
def trajectories(t_upper=3600*24*687, h=100):

    """Return the coordinates of the trajectories of three bodies
    
    Parameters
    ----------
    t_upper : scalar, the upper bound of time.  
        default: t_upper = 3600*24*687
    

    h : scalar, the step size for RK4 algorithm.
        default: h = 100 (Not recomended to change,  it can make the algorithm instable)


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
    # x_i1, ..., vz_i1 := p1('parametres')  # Pareil pour p2 et p3
    
    x_i1 = 1.0*1.496e+11       # initial values for planet 1 in x, y and z direction
    y_i1 = 0
    v_x1i = 0
    v_y1i = 29779.301841746023        
    z_i1 = 0
    v_z1i = 0

    x_i2 = 1.52*1.496e+11      # initial values for planet 2 in x, y and z direction
    y_i2 = 0
    v_x2i = 0
    v_y2i = 24154.203325249873
    z_i2 = 0
    v_z2i = 0  


    x_i3 = 0                    # initial values for Sun in x, y and z direction
    y_i3 = 0
    v_x3i = 0   
    v_y3i = 0  
    z_i3 = 0   
    v_z3i = 0   


    r = np.array([x_i1, y_i1, v_x1i, v_y1i, x_i2,
    y_i2, v_x2i, v_y2i, x_i3, y_i3, v_x3i, v_y3i, 
    z_i1,z_i2,z_i3,v_z1i,v_z2i,v_z3i])     # Initial positions and velocities


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




def visualisation(nbr_images, x1, y1, z1,
                               x2, y2, z2,
                               x3, y3, z3):

    # On vire t_max et on met les 9 trajectroires x_pts1, ..., 
    # Remplace nnr_images par 60

    """

    Create figures of the trajectories and orbits 
    of each body and save them in Images directory


    Parameter:
    ----------

    t_max: scalar, the number of point to plot in the figure to reprensent the trajectories of each bodies

    x1: array of shape (n,1), x coordinate of body 1
    y2: array of shape (n,1), y coordinate of body 1
    z3: array of shape (n,1), z coordinate of body 1
    x1: array of shape (n,1), x coordinate of body 2
    y2: array of shape (n,1), y coordinate of body 2
    z3: array of shape (n,1), z coordinate of body 2
    x1: array of shape (n,1), x coordinate of body 3
    y1: array of shape (n,1), y coordinate of body 3
    z1: array of shape (n,1), z coordinate of body 3


    """

    nbr_images = round(len(x_pnts1)/nbr_images)
    t_max = len(x_pnts1)
    i = 0

    # We create some random points to represent stars
    X = np.random.randint(-4e+11, 4e+11, 50, dtype=np.int64)
    Y = np.random.randint(-4e+11, 4e+11, 50, dtype=np.int64)
    Z = np.random.randint(-5000*3, 5000*3, 50, dtype=np.int64)

    l_month = np.arange(0, 23)

    for t in range(0, t_max, nbr_images):

        # We create a 3D figure

        fig = plt.figure()
        ax = p3.Axes3D(fig)


        ax.set_axis_off()
        # ax.set_facecolor('#202321')
        # ax.set_facecolor('#08143E')
        ax.set_facecolor('#071235')

        ax.view_init(elev=18)

        ax.get_proj = lambda: np.dot(p3.Axes3D.get_proj(ax), 
                      np.diag([1.6, 1.6, 0.9, 1]))

       
        # We create a circle of radius 1.52*1.496e+11, which
        # correspond to the distance of mars from sun

        # plane = Circle((0,0,0), 1.52*1.496e+11)
        # Circle.set_color(plane,'0.9')
        # Circle.set_alpha(plane, 0.1)


        # ax.add_patch(plane)

        # art3d.pathpatch_2d_to_3d(plane, z=0, zdir="z")
        

        # We plot the stars in different colors
        ax.scatter(X, Y, Z, s=0.1, marker='x', c='white')
        ax.scatter(0.5*X, 0.2*Y, 0.3*Z, s=0.2, marker='x', c='darkred')
        ax.scatter(-1*X, -1*Y, -1*Z, s=0.2, marker='x', c='darkorange')
        ax.scatter(-1*X, -0.5*Y, Z, s=0.2, marker='x', c='purple')
        

        ax.xaxis.set_pane_color((0.06, 0.06, 0.06, 0.99))
        ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 0.99))
        ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 0.99))


        # We plot a trajectories of each body

        ax.plot3D(x_pnts1, y_pnts1, z_pnts1, 'white', linewidth=1, alpha=1)
        ax.plot3D(x_pnts2, y_pnts2, z_pnts2, 'white', linewidth=1, alpha=1)

        ax.plot3D(x_pnts1, y_pnts1, z_pnts1, 'cornflowerblue', linewidth=0.9)
        ax.plot3D(x_pnts2, y_pnts2, z_pnts2, 'tomato', linewidth=0.9)
        

        # We make the grid lines transparent

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        # Set x, y and z limits

        ax.set_xlim3d([-2.5e+11, 2.5e+11])
        ax.set_xlabel('X')

        ax.set_ylim3d([-2.5e+11, 2.5e+11])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-5000, 5000])
        ax.set_zlabel('Z')


        # We create a ball that represents the sun

        ax.plot3D(0,0,0, 'o', markersize=40, color='darkorange')


        # We plot the postion of each bodies at each time t

        ax.plot(x_pnts1[t], y_pnts1[t], z_pnts1[t], 'o',
                 markersize=7, color='darkblue')  
        ax.plot(x_pnts2[t], y_pnts2[t], z_pnts2[t], 'o', 
                markersize=5, color='darkred')  

        for time in l_month:

            if time*(len(x_pnts1)/23) <= t <= (time+1)*(len(x_pnts1)/23):

                tm = 'Elapsed time : ' + str(l_month[time]) + ' Earth month'
                ax.text(0.5e+11, 0, 11000, tm, c='black',
                         fontsize='small',
                         bbox=dict(facecolor='white', edgecolor='white',
                         boxstyle='round,pad=0.6'))
 

        # We make some pause

        plt.pause(0.00000001) 


        # We save the figure as png file to make gif later

        plt.savefig('threebody/EDO/Images2/file_{:03}.png'.format(i))


        i += 1

        # We close the figure to make the next one
        plt.close()


# x_pnts1, y_pnts1, x_pnts2, y_pnts2, x_pnts3, y_pnts3, z_pnts1, z_pnts2, z_pnts3 = trajectories()
# visualisation(nbr_images=80, x1=x_pnts1, y1=y_pnts1, z1=z_pnts1,
                                # x2=x_pnts2, y2=y_pnts2, z2=z_pnts2,
                                # x3=x_pnts3, y3=y_pnts3, z3=z_pnts3)


class Animation():

    """ 

    This class was made to create gif to represent 
    tajectories of our three bodies in 3D dimension,
    from an image list of png.

    """

    def __init__(self):

        """ 

        We specify to the user that the instance was created
        and that he is about to create a gif
        
        """

        print("Instance Created, you will create a gif.") 



    def __call__(self, image_path, fps, gif_path):


        """ 

        This function create a gif from a list of png file

        Parameters
        -----------

        image_path: path, indicates where to find the list of png files to create the gif

        fps: integer, the fps (frames per second) for the gif

        gif_path: path, where to put the gif created


        Example
        ---------

        gif = CreateGif() --> create an instance of the class CreateGif

        gif(image_path=Path('threebody/EDO/Images'), fps=15,
        gif_path=Path('threebody/EDO/Three_body.gif')) --> will be create a gif from the list
        of png located at threebody/EDO/Three_body.gif, with 15 frames per second,
        and the gif will be put at threebody/EDO/ under the name Three_body.gif


        """

        self.image_path = image_path
        self.fps = fps
        self.gif_path = gif_path
        self.images = list(self.image_path.glob('*.png'))
        self.image_list = []


        # We test if the path where the png files are located exist or not
        # If not, we raise an error

        try: 
            assert os.path.exists(self.image_path)

        except Exception as error:
            print("Caught an exception:", error)


        # We check if the list wich contain the png file 
        # is not empty

        try: 
            assert len(self.images) != 0

        except Exception as error:
            print("Caught an exception:", error)

        
        # We test if the fps parameter is an integer or not
        # If not, we raise an error
        
        try:
            assert isinstance(self.fps, int)

        except Exception as error:
            print("Caught an exception:", error)


        # We make a loop to read the png file
        # and add them in image list

        for file_name in self.images:

            self.image_list.append(imageio.imread(file_name))


        # We create a gif and save them in gif_path path

        imageio.mimsave(self.gif_path, self.image_list, fps=self.fps)



# We create an instance of the class CreateGif
gif = Animation()

# __call__ method will be called here
gif(image_path=Path('threebody/EDO/Images2'), fps=11,
    gif_path=Path('threebody/EDO/Three_body2.gif'))



