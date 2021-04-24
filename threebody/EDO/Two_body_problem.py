# The Two-Body Model

# It's a star system.
# It contains three stars - Alpha Centauri A, Alpha Centauri B and
# Alpha Centauri C However, since Proxima Centauri has negligible
# mass compared to the other two stars, Alpha Centauri is
# considered a system of binary stars.

# %%
import scipy.integrate
import scipy as sci
import matplotlib as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# Define the constants and the quantities

G = 6.67408e-11  # Universal gravitational constant N-m2 / kg2
m_nd = 1.989e+30  # mass of the sun (kg)
r_nd = 5.326e+12  # distance between stars in Alpha Centauri (m)
v_nd = 30000      # relative speed of the earth around the sun (m / s)
t_nd = 79.91*365*24*3600*0.51  # orbital period of Alpha Centauri (in sedond s)

# The clear constants
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd


# Define some parameters that define the two stars
# that we are trying to simulate - their masses

m1 = 1.1  # Alpha Centauri A, ndicating 1.1 times the mass of the sun, which is our refer
m2 = 0.907  # Alpha Centauri B
r1 = [- 0.5, 0, 0]  # m Initial position vectors
r2 = [0.5, 0, 0]  # m Initial position vectors

r1 = sci.array(r1, dtype="float64")  # Convert position vectors to arrays
r2 = sci.array(r2, dtype="float64")
r_com = (m1*r1 + m2*r2)/(m1 + m2)  # Find the center of mass
v1 = [0.01, 0.01, 0]  # m/s initial speeds
v2 = [-0.05, 0, -0.1]  # m/s initial speeds

v1 = sci.array(v1, dtype="float64")  # Convert Velocity Vectors to Arrays
v2 = sci.array(v2, dtype="float64")

v_com = (m1*v1 + m2*v2)/(m1 + m2)  # Find COM speed


def TwoBodyEquations(w, t, G, m1, m2):

    """
        This function takes in an array containing all the dependent variables
        (here the position and the speed) and an array containing all the
        independent variables (here the time) in that order.
        It returns the values of all the differentials in an array.
    """

    r1 = w[: 3]
    r2 = w[3: 6]
    v1 = w[6: 9]
    v2 = w[9: 12]
    r = sci.linalg.norm(r2-r1)  # Calculate the norm of the vector
    dv1bydt = K1 * m2 * (r2-r1) / r ** 3
    dv2bydt = K1 * m1 * (r1-r2) / r ** 3
    dr1bydt = K2 * v1
    dr2bydt = K2 * v2
    r_derivs = sci.concatenate((dr1bydt, dr2bydt))
    derives = sci.concatenate((r_derivs, dv1bydt, dv2bydt))

    return derives


# We define the function, the initial conditions and the time interval in the odeint function

init_params = sci.array([r1, r2, v1, v2])  # create array of initial params
init_params = init_params.flatten()  # flatten array to make it 1D
time_span = sci.linspace(0, 8, 500)  # 8 orbital periods and 500 points

# Run the ODE solver
two_body_sol = sci.integrate.odeint(TwoBodyEquations, init_params, time_span, args=(G, m1, m2))


r1_sol = two_body_sol[:, : 3]
r2_sol = two_body_sol[:, 3: 6]

# Find the location of COM
rcom_sol = (m1*r1_sol + m2*r2_sol)/(m1 + m2)

# Find the location of Alpha Centauri A by COM
r1com_sol = r1_sol-rcom_sol

# Find the location of Alpha Centauri B by COM
r2com_sol = r2_sol-rcom_sol

# Create figure
fig = plt.figure(figsize=(15, 15))

# Create 3D axes
ax = fig.add_subplot(111, projection="3d")

# Draw the orbits
ax.plot(r1com_sol[:, 0], r1com_sol[:, 1], r1com_sol[:, 2], color="darkblue")
ax.plot(r2com_sol[:, 0], r2com_sol[:, 1], r2com_sol[:, 2], color="green")

# Draw the final positions of the stars
ax.scatter(r1com_sol[-1, 0], r1com_sol[-1, 1], r1com_sol[-1, 2], color="darkblue", marker="o", s=100, label="Alpha Centauri A")
ax.scatter(r2com_sol[-1, 0], r2com_sol[-1, 1], r2com_sol[-1, 2], color="green", marker="o", s=100, label=" Alpha Centauri B ")

# Ajoutez quelques cloches et sifflets supplémentaires
ax.set_xlabel("coordonnée x", fontsize=14)
ax.set_ylabel("coordonnée y", fontsize=14)
ax.set_zlabel("coordonnée z", fontsize=14)
ax.set_title("Visualization of the orbits of stars in a two-body system", fontsize=14)
ax.legend(loc="upper left", fontsize=14)
