Exlanation functions
======================

All the functions of threebody package
---------------------------------------



.. automodule:: threebody.EDO

.. autofunction:: EDO.EDO_3.distance

.. note:: 
	The formula used is the euclidian distance, given below
.. math:: d(X,Y) = \sqrt{\sum_{i=1}^{n} (x_i-y_i)^2}

.. autofunction:: EDO.EDO_3.velocity
.. note::
        The formula used is for the velocity `v` is given below
.. math:: v = \sqrt{\frac{G*M}{r}}


.. autofunction:: EDO.EDO_3.f

`Method used:`

This function uses the `RK4` (Runge Kutta 4) method to solve 
the differential system composed of 18 equations of order 1 
of the 3 body problem.
The RK4 method allows to solve the differential equation `y' = f(y,t)`,
where y and t can be a scalar or vector and y' is the derivative of y.
In physic's problem, t represent generally the time, this is the case for three body problem
    

.. autofunction:: EDO.EDO_3.trajectories
.. note::  
	t_upper is put at 24*3600*687 to simulate a marsian year.
    	It can be put at 24*3600*365 to simulate a earth year. 


.. Attention:: 
	h is not recomended to change, it can make the algorithm very instable


.. autofunction:: Vis.visualisation.visualisation

.. autoclass:: Vis.visualisation.Animation
   :members: __call__

Saturne_Mercure_Sun_problem's function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: EDO.Saturne_Mercure_Sun.init

.. autofunction:: EDO.Saturne_Mercure_Sun.Force_S_Sun

.. autofunction:: EDO.Saturne_Mercure_Sun.Force_M_Sun

.. autofunction:: EDO.Saturne_Mercure_Sun.Force_S_M

.. autofunction:: EDO.Saturne_Mercure_Sun.Force

.. autofunction:: EDO.Saturne_Mercure_Sun.Euler


--------------------------------------------------------------------

Two_body_problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction EDO.Two_body_problem.TwoBodyEquations

.. note::
         TwoBodyEquations
	 This function takes in an array containing all the dependent variables
	 (here the position and the speed) and an array containing all the
	 independent variables (here the time) in that order. It returns the values
	 of all the differentials in an array
