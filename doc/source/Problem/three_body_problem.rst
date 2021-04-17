Three body problem
====================


The problem that we are trying to modelize is the three-body problem.
The three-body problem consist in taking the initial positions and velocities of
three bodies (planets, stars, ...) and then solving for their mutual motion according 
to Newton's laws of motion and Newton's law of universal gravitation.

.. math::
  
   {\displaystyle {\begin{aligned}{\ddot {\mathbf {r} }}_{\mathbf {1} }
   &=-Gm_{2}{\frac {\mathbf {r_{1}} -\mathbf {r_{2}} }{|\mathbf {r_{1}} -\mathbf {r_{2}} |^{3}}}-Gm_{3}{\frac {\mathbf {r_{1}} -\mathbf {r_{3}} }
   {|\mathbf {r_{1}} -\mathbf {r_{3}} |^{3}}},\\{\ddot {\mathbf {r} }}_{\mathbf {2} }
   &=-Gm_{3}{\frac {\mathbf {r_{2}} -\mathbf {r_{3}} }{|\mathbf {r_{2}} -\mathbf {r_{3}} |^{3}}}-Gm_{1}{\frac {\mathbf {r_{2}} -\mathbf {r_{1}} }
   {|\mathbf {r_{2}} -\mathbf {r_{1}} |^{3}}},\\{\ddot {\mathbf {r} }}_{\mathbf {3} }&=-Gm_{1}{\frac {\mathbf {r_{3}} -\mathbf {r_{1}} }
   {|\mathbf {r_{3}} -\mathbf {r_{1}} |^{3}}}-Gm_{2}{\frac {\mathbf {r_{3}} -\mathbf {r_{2}} }{|\mathbf {r_{3}} -\mathbf {r_{2}} |^{3}}}.\end{aligned}}}


Source: https://en.wikipedia.org/wiki/Three-body_problem
