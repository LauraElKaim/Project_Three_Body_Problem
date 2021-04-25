Two body problem
=================

In classical mechanics, the two-body problem is to predict the motion of 
two massive objects which are abstractly viewed as point particles. The problem 
assumes that the two objects attract each other, so the only force 
affecting each object arises from the other one, and all other objects are ignored.  

Perhaps a famous real-world example of a two-body system is the Alpha Centauri 
star system. It contains three stars - Alpha Centauri A, Alpha Centauri B and 
Alpha Centauri C (commonly referred to as Proxima Centauri). However, since 
Proxima Centauri has negligible mass compared to the other two stars, Alpha 
Centauri is considered a binary star system.  

An important point to note here is 
that the bodies considered in an `n`-body system all have similar masses. Thus, 
Sun-Earth-Moon is not a three-body system because they do not have equivalent 
masses and the Earth and the Moon do not significantly influence the path of the Sun.

The equation below represents this law in vector form:  
 .. math::
   
    \overrightarrow{F} = \frac{Gm_1m_2}{r^2}\widehat{r}

Thus, we have the movement equation:  

.. math::

    m_1\frac{d{^2} \overrightarrow{r_1}}{dt^2} = \frac{Gm_1m_2}{r^3}r_{12} 
  
We have a second order differential equation that describes the interaction between two bodies due to gravity. To simplify its solution, 
we can decompose it into two first-order differential equations.

The equation of speed is given the speed as a first order 
differential of the position.

.. math::
        
    m_1\frac{d\overrightarrow{v_i}}{dt}=\frac{Gm_im_j}{r_{ij}^3}r_{ij}

.. math::

    \frac{d\overrightarrow{v_i}}{dt} = v_i 

The index :math:`i` is for the body whose position and velocity are to be calculated 
while the index :math:`j` is for the other body that interacts with body :math:`i`. Thus, 
we will solve two sets of these two equations for the two-body system.