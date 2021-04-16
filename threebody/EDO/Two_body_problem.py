Two_body_problem

## The Two-Body Model

#Système d'étoiles Alpha Centauri
#Perhaps a famous real-world example of a two-body system is the Alpha Centauri star system.
#It contains three stars - Alpha Centauri A, Alpha Centauri B and Alpha Centauri C (commonly referred to as Proxima Centauri).
#However, since Proxima Centauri has negligible mass compared to the other two stars, Alpha Centauri is considered a system of binary stars.  

#%%
#Import scipy
import scipy as sci
import scipy as sp
#Import matplotlib and associated modules for 3D and animations
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
%matplotlib notebook


# We are going to define the constants and the quantities of reference which make it possible not to dimension 
#the equations as well as the net constants K₁ and K₂.

#%%
#Définir la constante de gravitation universelle 
G = 0.0000000067 # N-m2 / kg2
# Quantités de référence 
m_nd = 1.989e+30 #kg # masse du soleil 
r_nd = 5.326e+12 #m # distance entre les étoiles dans Alpha Centauri 
v_nd = 30000 # m / s # vitesse relative de la terre autour du soleil 
t_nd = 79.91 * 365 * 24 * 3600 * 0.51 #s # période orbitale d'Alpha Centauri
# Constantes nettes 
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd


# We are going to define some parameters that define the two stars that we are trying to simulate - their masses,
#their initial positions and their initial speeds. These parameters are not dimensional, so the mass of Alpha Centauri A 
#is defined as 1.1 (indicating 1.1 times the mass of the sun, which is our reference quantity). The speeds are arbitrarily 
#defined so that none of the bodies escapes the gravitational attraction of each other.


#%%
import scipy as sp
#Définir les masses 
m1 = 1.1 #Alpha Centauri A 
m2 = 0.907 #Alpha Centauri B
#Définir les vecteurs de position initiale 
r1 = [- 0.5,0,0] #m 
r2 = [0.5,0,0] #m
#Convertir les vecteurs pos en tableaux 
r1 = sci.array (r1, dtype = "float64") 
r2 = sci.array (r2, dtype = "float64")
# Trouver le centre de masse 
r_com = (m1 * r1 + m2 * r2) / (m1 + m2)
#Définir les vitesses initiales 
v1 = [0.01,0.01,0] # m / s 
v2 = [- 0.05,0, -0.1] # m / s
#Convertir les vecteurs de vitesse en tableaux 
v1 = sci.array (v1, dtype = "float64") 
v2 = sci.array (v2, dtype = "float64")
# Trouver la vitesse de COM 
v_com = (m1 * v1 + m2 * v2) / (m1 + m2)

#%%

# %%
def TwoBodyEquations (w, t, G, m1, m2): 
    
    r1 = w [: 3] 
    r2 = w [3: 6] 
    v1 = w [6: 9] 
    v2 = w [9 : 12]
    r = sci.linalg.norm (r2-r1) #Calculer la magnitude ou la norme du vecteur
    dv1bydt = K1 * m2 * (r2-r1) / r ** 3 
    dv2bydt = K1 * m1 * (r1-r2) / r ** 3 
    dr1bydt = K2 * v1 
    dr2bydt = K2 * v2
    r_derivs = sci.concatenate ((dr1bydt, dr2bydt)) 
    dérivés = sci.concatenate ((r_derivs, dv1bydt, dv2bydt)) 
    return  dérivés

# We define the function, the initial conditions and the time interval in the odeint function    

#%%
init_params = sci.array ([r1, r2, v1, v2]) #create array of initial params 
init_params = init_params.flatten () #flatten array pour le rendre 1D 
time_span = sci.linspace (0,8,500) # 8 périodes orbitales et 500 points
#Exécuter le solveur ODE 
import scipy.integrate
two_body_sol = sci.integrate.odeint (TwoBodyEquations, init_params, time_span, args = (G, m1, m2))


# %%
r1_sol = two_body_sol [:,: 3] 
r2_sol = two_body_sol [:, 3: 6]

# %%
# Trouver l'emplacement de COM 
rcom_sol = (m1 * r1_sol + m2 * r2_sol) / (m1 + m2)
# Trouver l'emplacement d'Alpha Centauri A wrt COM 
r1com_sol = r1_sol-rcom_sol
# Trouver l'emplacement d'Alpha Centauri B par COM 
r2com_sol = r2_sol-rcom_sol

# %%

#Create figure 
fig = plt.figure (figsize = (15,15))
#Créer des axes 3D 
ax = fig.add_subplot (111, projection = "3d")
# Tracez les orbites 
ax.plot (r1com_sol [:, 0], r1com_sol [:, 1], r1com_sol [:, 2], color = "darkblue") 
ax.plot (r2com_sol [:, 0], r2com_sol [:, 1], r2com_sol [:, 2], color = "red")
# Tracez les positions finales des étoiles 
ax.scatter (r1com_sol [-1,0], r1com_sol [-1,1], r1com_sol [-1,2], color = "darkblue", marker = "o", s = 100, label = "Alpha Centauri A") 
ax.scatter (r2com_sol [-1,0], r2com_sol [-1,1], r2com_sol [-1,2], color = "red", marker = "o", s = 100, label =" Alpha Centauri B ")
# Ajoutez quelques cloches et sifflets supplémentaires 
ax.set_xlabel ("coordonnée x", fontsize = 14) 
ax.set_ylabel ("coordonnée y", fontsize = 14) 
ax.set_zlabel ("coordonnée z", fontsize = 14) 
ax.set_title ("Visualisation des orbites des étoiles dans un système à deux corps \ n", fontsize = 14) 
ax.legend (loc = "upper left", fontsize = 14)
# %%
