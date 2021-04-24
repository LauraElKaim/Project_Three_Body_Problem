from matplotlib import pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from threebody.EDO.EDO_3 import trajectories
# from IPython.display import Video
import moviepy.editor as mp
# from IPython.display import Video


class Mars():

    """

    This class represent Mars.
    They give all the parameters for solving three body problem
    when considering the Sun at the center of space.

    """

    def init(self):

        """All parameters of Mars needed to solve for three body problem"""

        self.mass = 6.417e+23
        self.velocity_x = 0
        self.velocity_y = 0
        self.velocity_z = 0
        self.postion_x = 1.52*1.496e+11
        self.postion_y = 0
        self.postion_z = 0

    def str(self):

        return f"""                The Mars'mass: {self.mass} \n
                The velocity'mass in x direction: {self.velocity_x} \n
                The velocity'mass in y direction: {self.velocity_y} \n
                The velocity'mass in y direction: {self.velocity_y} \n
                The velocity'mass in z in direction: {self.velocity_z} \n
                The Mars'position in x direction: {self.postion_x} \n
                The Mars'position in y direction: {self.postion_y} \n
                The Mars'position in z direction: {self.postion_z}"""


def Mars_Orbit(save_anim=False, fps=24, save_as_mp4=False):

    x_pnts1, y_pnts1, x_pnts2, y_pnts2, x_pnts3, y_pnts3, z_pnts1, z_pnts2, z_pnts3 = trajectories(h=100)

    # Create figure 3D
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # We create some random points to represent stars
    X = np.random.randint(-4e+11, 4e+11, 50, dtype=np.int64)
    Y = np.random.randint(-4e+11, 4e+11, 50, dtype=np.int64)
    Z = np.random.randint(-5000*3, 5000*3, 50, dtype=np.int64)

    # Plot some stars
    ax.scatter(X, Y, Z, s=0.05, marker='x', c='white')
    ax.scatter(0.5*X, 0.2*Y, 0.3*Z, s=0.05, marker='x', c='darkred')
    ax.scatter(-1*X, -1*Y, -1*Z, s=0.05, marker='x', c='darkorange')
    ax.scatter(-1*X, -0.5*Y, Z, s=0.05, marker='x', c='purple')

    # We make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    # set the parameters for the visual
    ax.set_axis_off()
    ax.set_facecolor('#070E25')
    ax.view_init(elev=32)

    def gen(n, x, y, z):

        """Generator to browse the points of the trajectory vectors"""

        t = 0

        while t < len(x):
            yield np.array([x[t], y[t], z[t]])
            t += n

    def update(num, data, line):

        """Set line"""

        line.set_data(data[:2, :num])
        line.set_3d_properties(data[2, :num])

    N = 3000
    data = np.array(list(gen(n=N, x=x_pnts2, y=y_pnts2, z=z_pnts2))).T
    line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1],
                    c='red', linestyle=":")

    ax.plot3D(0, 0, 0, 'o', markersize=50, color='darkorange')

    ax.text(-4.5e+11, 0, 7500, 'Sun', c='white', fontsize=8,
            bbox=dict(facecolor='darkorange', edgecolor='white',
                      boxstyle='circle,pad=0.5'))

    ax.text(-3.8e+11, 0, 7900, 'Mars', c='white', fontsize=6,
            bbox=dict(facecolor='red', edgecolor='white',
                      boxstyle='circle,pad=0.3'))

    # Set x, y and z limits
    ax.set_xlim3d([-2.5e+11, 2.5e+11])
    ax.set_xlabel('X')

    ax.set_ylim3d([-2.5e+11, 2.5e+11])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-5000, 5000])
    ax.set_zlabel('Z')

    animation_Mars = animation.FuncAnimation(fig, update,
                                             frames=200, fargs=(data, line),
                                             interval=10, blit=False)

    # save the gif if save_anim is True
    if save_anim:

        animation_Mars.save('Mars_orbit.gif', fps=fps, writer='ffmpeg')

    # save the gif as mp4 if save_as_mp4 is True
    if save_as_mp4:

        clip = mp.VideoFileClip("Mars_orbit.gif")
        clip.write_videofile("myvideo.mp4")

    plt.show()


Mars_Orbit(save_anim=True, fps=24)
