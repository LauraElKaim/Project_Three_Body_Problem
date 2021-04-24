Here are some examples of how to use the package threebody
===========================================================

If you want to solve for threebody problem of body 1, body 2, body 3, with respectively
mass M1, M2, M3 (and we can assume that the third planet is in the center of space),
you can simply do this as follows.

.. code-block:: python
            
    from threebody.EDO.EDO_3 import distance
    from threebody.EDO.EDO_3 import trajectories

    X1 = [x1, y1, z1]  # coordiante of body one when body one is in the center of space
    X2 = [x2, y2, z2]  # coordiante of body one when body two is in the center of space
    
    d1 = distance(X1, X3)   # distance between body one and body 3
    d2 = distance(Y1, Y3)  # distance between body two and body 3
    
    x1, y1, x2, y2, x3, y3, z1, z2, z3 = trajectories(h=100, m1=M1, m2=M2, m3=M3, a1=d1, a2=d2)

If you want to generate some images and then create a gif, you can do this as follows.

.. code-block:: python

    from threebody.EDO.EDO_3 import trajectories
    from threebody.Vis.visualisation import visualisation, Animation
    from path import Path
    import moviepy.editor as mp

    # We call trajectories function to calculate the trajectories
    # x1, y1, x2, y2, x3, y3, z1, z2, z3 = trajectories(h=100)

    # We call the visualisation function to create some images
    visualisation(nbr_images=200, save_path='path/where/to/save/images',
                    x1=x1, y1=y1, z1=z1,
                    x2=x2, y2=y2, z2=z2,
                    x3=x3, y3=y3, z3=z3)

        
    # Create a gif from list of png file
    gif = Animation()
    gif(image_path=Path('path/where/are/located/images'), fps=40,
        gif_path=Path('path/where/to/save/my_gif.gif'))

   
    # Convert a gif created to mp4 file
    clip = mp.VideoFileClip("path/to/my_gif.gif")
    clip.write_videofile("path/where/to/save/my_video.mp4")
