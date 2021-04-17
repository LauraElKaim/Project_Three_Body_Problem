from threebody.EDO.EDO_3 import trajectories
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as p3
import imageio
from matplotlib.patches import FancyArrowPatch, Circle
import mpl_toolkits.mplot3d.art3d as art3d
from path import Path
import time
import os.path



def visualisation(nbr_images, save_path, x1, y1, z1,
                    x2, y2, z2, x3, y3, z3):

    """Create figures of the trajectories and orbits
    of each body and save them in Images directory


    Parameters
    ----------

    t_max: scalar,
        The number of point to plot in the figure to reprensent
        the trajectories of each bodies

    x1: array of shape (n, 1),
        x coordinate of body 1
    y1: array of shape (n, 1),
        y coordinate of body 1
    z1: array of shape (n, 1),
        z coordinate of body 1
    x2: array of shape (n, 1),
        x coordinate of body 2
    y2: array of shape (n, 1),
        y coordinate of body 2
    z2: array of shape (n, 1),
        z coordinate of body 2
    x3: array of shape (n, 1),
        x coordinate of body 3
    y3: array of shape (n, 1),
        y coordinate of body 3
    z3: array of shape (n, 1),
        z coordinate of body 3

    save_path: string, default:'.' (current directory)
        path in which you want to save the images
    """

    nbr_images = round(len(x1)/nbr_images)
    t_max = len(x1)
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
        ax.scatter(0.5*X, 0.2*Y, 0.3*Z, s=0.1, marker='x', c='darkred')
        ax.scatter(-1*X, -1*Y, -1*Z, s=0.1, marker='x', c='darkorange')
        ax.scatter(-1*X, -0.5*Y, Z, s=0.1, marker='x', c='purple')


        ax.xaxis.set_pane_color((0.06, 0.06, 0.06, 0.99))
        ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 0.99))
        ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 0.99))


        # We plot a trajectories of each body
        ax.plot3D(x1, y1, z1, 'white', linewidth=1, alpha=1)
        ax.plot3D(x2, y2, z2, 'white', linewidth=1, alpha=1)

        ax.plot3D(x1, y1, z1, 'cornflowerblue', linewidth=0.9)
        ax.plot3D(x2, y2, z2, 'tomato', linewidth=0.9)


        # We make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0)

        # Set x, y and z limits
        ax.set_xlim3d([-2.5e+11, 2.5e+11])
        ax.set_xlabel('X')

        ax.set_ylim3d([-2.5e+11, 2.5e+11])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-5000, 5000])
        ax.set_zlabel('Z')


        # We create a ball that represents the sun
        ax.plot3D(0, 0, 0, 'o', markersize=50, color='darkorange')


        # We plot the postion of each bodies at each time t
        ax.plot(x1[t], y1[t], z1[t], 'o',
                markersize=7, color='darkblue')
        ax.plot(x2[t], y2[t], z2[t], 'o', 
                markersize=5, color='darkred')  

        for t_month in l_month:

            if t_month*(len(x1)/23) <= t <= (t_month+1)*(len(x1)/23):

                tm = 'Elapsed time : ' + str(l_month[t_month]) + ' Earth month'
                ax.text(0.6e+11, 0, 11500, tm, c='black',  fontsize=8, va='top',
                        bbox=dict(facecolor='white', edgecolor='white',
                        boxstyle='round,pad=0.6'))


        #Add legend
        ax.text(-2.7e+11, 0, 9000, 'Sun', c='white', fontsize=8,
                        bbox=dict(facecolor='darkorange', edgecolor='white',
                        boxstyle='circle,pad=0.5'))


        ax.text(-2.2e+11, 0, 9200, 'Earth', c='white', fontsize=6,
                        bbox=dict(facecolor='darkblue', edgecolor='white',
                        boxstyle='circle,pad=0.3'))


        ax.text(-1.7e+11, 0, 9400, 'Mars', c='white', fontsize=6,
                        bbox=dict(facecolor='darkred', edgecolor='white',
                        boxstyle='circle,pad=0.3'))


        # We make some pause
        plt.pause(0.00000001)

        # We save the figure as png file to make gif later
        plt.savefig(save_path + '/file_{:03}.png'.format(i),
                    bbox_inches = 'tight', pad_inches = 0)

        i += 1

        # We close the figure to make the next one
        plt.close()



class Animation():

    """This class was made to create gif to represent
    tajectories of our three bodies in 3D dimension,
    from an image list of png.
    """

    def __init__(self):

        """We specify to the user that the instance was created
        and that he is about to create a gif
        """

        print("Instance Created, you will create a gif.")



    def __call__(self, image_path, fps, gif_path):


        """This function create a gif from a list of png file

        Parameters
        -----------

        image_path: string (path),
            indicates where to find the list of png files to create the gif

        fps: integer,
            the fps (frames per second) for the gif

        gif_path: string (path),
            where to put the gif created


        Example
        ---------

        >>> from threebody import Animation
        >>> gif = Animationf()
        >>> gif(image_path=Path('threebody/EDO/Images'), fps=15,
        gif_path=Path('threebody/EDO/Three_body.gif'))
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


x_pnts1, y_pnts1, x_pnts2, y_pnts2, x_pnts3, y_pnts3, z_pnts1, z_pnts2, z_pnts3 = trajectories(h=100)
visualisation(nbr_images=80, save_path='threebody/Vis/Image',
                x1=x_pnts1, y1=y_pnts1, z1=z_pnts1,
                x2=x_pnts2, y2=y_pnts2, z2=z_pnts2,
                x3=x_pnts3, y3=y_pnts3, z3=z_pnts3)

# gif = Animation()
# gif(image_path=Path('threebody/Vis/Image'), fps=10,
#     gif_path=Path('threebody/Vis/Three_body.gif'))
