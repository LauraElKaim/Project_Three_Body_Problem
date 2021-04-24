from threebody.EDO.EDO_3 import trajectories
from threebody.Vis.visualisation import visualisation
from threebody.Vis.visualisation import Animation
from path import Path
import moviepy.editor as mp


# We call trajectories function to calculate the trajectories
x_pnts1, y_pnts1, x_pnts2, y_pnts2, x_pnts3, y_pnts3, z_pnts1, z_pnts2, z_pnts3 = trajectories(h=100)


# We call the visualisation function to create some images
visualisation(nbr_images=200, save_path='threebody/Vis/Images/Images_2',
              x1=x_pnts1, y1=y_pnts1, z1=z_pnts1,
              x2=x_pnts2, y2=y_pnts2, z2=z_pnts2,
              x3=x_pnts3, y3=y_pnts3, z3=z_pnts3)


# Create a gif from list of png file
# gif = Animation()
# gif(image_path=Path('threebody/Vis/Images/Images_2'), fps=40,
#     gif_path=Path('threebody/Vis/Gifs/Three_body.gif'))

# Convert a gif created to mp4 file
# clip = mp.VideoFileClip("threebody/Vis/Gifs/Three_body.gif")
# clip.write_videofile("./myvideo.mp4")
