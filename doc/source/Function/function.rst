Explanation functions
======================

All the functions of threebody package
---------------------------------------

EDO_3's functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: threebody.EDO.EDO_3.distance

.. note:: 
	The formula used is the euclidian distance, given below
.. math:: d(X,Y) = \sqrt{\sum_{i=1}^{n} (x_i-y_i)^2}

.. autofunction:: threebody.EDO.EDO_3.velocity

.. note::
        The formula used is for the velocity :math:`v` is given below
.. math:: v = \sqrt{\frac{G*M}{r}}


.. autofunction:: threebody.EDO.EDO_3.trajectories

`Method used:`

This function uses the `RK4` (Runge Kutta 4) method to solve 
the differential system composed of 18 equations of order 1 
of the 3 body problem.
The `RK4` method allows to solve the differential equation :math:`y' = f(y,t)`,
where :math:`y` and :math:`t` can be a scalar or vector and :math:`y'` is the derivative of :math:`y`.
In physic's problem, :math:`t` represent generally the time, this is the case for three body problem.   

.. note::  
	`t_upper` is put at 24*3600*687 to simulate a marsian year.
    	It can be put at 24*3600*365 to simulate a earth year. 


.. Attention:: 
	:math:`h` is not recomended to change, it can make the algorithm very instable

.. autofunction:: threebody.EDO.EDO_3.derivative

.. autofunction:: threebody.Vis.visualisation.visualisation


.. autoclass:: threebody.Vis.visualisation.Animation
   :members: __call__


.. autofunction:: threebody.Vis.visualisation.ConvertGif


Saturne_Mercure_Sun_problem's function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: threebody.EDO.Saturne_Mercure_Sun.init

.. autofunction:: threebody.EDO.Saturne_Mercure_Sun.Force_S_Sun

.. autofunction:: threebody.EDO.Saturne_Mercure_Sun.Force_M_Sun

.. autofunction:: threebody.EDO.Saturne_Mercure_Sun.Force_S_M

.. autofunction:: threebody.EDO.Saturne_Mercure_Sun.Force

.. autofunction:: threebody.EDO.Saturne_Mercure_Sun.Euler


--------------------------------------------------------------------

Two_body_problem's function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction threebody.EDO.Two_body_problem.TwoBodyEquations

.. note::
        TwoBodyEquations
	 	This function takes in an array containing all the dependent variables
		(here the position and the speed) and an array containing all the
	 	independent variables (here the time) in that order. It returns the values
	 	of all the differentials in an array
