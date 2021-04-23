Three body problem
====================


The problem that we are trying to modelize is the three-body problem.
The three-body problem consist in taking the initial positions and velocities of
three bodies (planets, stars, ...) and then solving for their mutual motion according 
to Newton's laws of motion and Newton's law of universal gravitation.

So if we note :math:`r_i = (x_i, y_i, z_i)` the position of the body :math:`i`, and and :math:`m_i` its mass, 
we need to solve the following differential system equation:   

.. math::
  
   {\displaystyle {\begin{aligned}{\ddot {\mathbf {r} }}_{\mathbf {1} }
   &=-Gm_{2}{\frac {\mathbf {r_{1}} -\mathbf {r_{2}} }{|\mathbf {r_{1}} -\mathbf {r_{2}} |^{3}}}-Gm_{3}{\frac {\mathbf {r_{1}} -\mathbf {r_{3}} }
   {|\mathbf {r_{1}} -\mathbf {r_{3}} |^{3}}},\\{\ddot {\mathbf {r} }}_{\mathbf {2} }
   &=-Gm_{3}{\frac {\mathbf {r_{2}} -\mathbf {r_{3}} }{|\mathbf {r_{2}} -\mathbf {r_{3}} |^{3}}}-Gm_{1}{\frac {\mathbf {r_{2}} -\mathbf {r_{1}} }
   {|\mathbf {r_{2}} -\mathbf {r_{1}} |^{3}}},\\{\ddot {\mathbf {r} }}_{\mathbf {3} }&=-Gm_{1}{\frac {\mathbf {r_{3}} -\mathbf {r_{1}} }
   {|\mathbf {r_{3}} -\mathbf {r_{1}} |^{3}}}-Gm_{2}{\frac {\mathbf {r_{3}} -\mathbf {r_{2}} }{|\mathbf {r_{3}} -\mathbf {r_{2}} |^{3}}},\end{aligned}}}

where :math:`G` is the gravitational constant (:math:`\approx 6,67.10^{-11} m^{3}⋅kg^{−1}⋅s^{−2}`) and the left hand side correspond to the second derivative of position :math:`r` with respect to time :math:`t`.

Source: https://en.wikipedia.org/wiki/Three-body_problem


.. note::
	 The equation system above, composed of 9 equations of order 2, 
	 is transformed into an equation system of 18 differentials equations 
    	 of order 1, so that it can be solved by the `RK4` method.

