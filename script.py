from threebody.EDO.EDO_3 import trajectories
from threebody.Vis.visualisation import visualisation, Animation
from path import Path


x_pnts1, y_pnts1, x_pnts2, y_pnts2, x_pnts3, y_pnts3, z_pnts1, z_pnts2, z_pnts3 = trajectories(h=100000)
visualisation(nbr_images=10, save_path='threebody/Vis/Tempory',
                x1=x_pnts1, y1=y_pnts1, z1=z_pnts1,
                x2=x_pnts2, y2=y_pnts2, z2=z_pnts2,
                x3=x_pnts3, y3=y_pnts3, z3=z_pnts3)


gif = Animation()
gif(image_path=Path('threebody/Vis/Tempory'), fps=40,
    gif_path=Path('threebody/Vis/Gifs/Three_body.gif'))
