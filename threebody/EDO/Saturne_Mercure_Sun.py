import math
import numpy as np
import pylab as py

from IPython.display import HTML

# We are interested in the problem of the three bodies
# for the Sun, Mercure and Saturne,
# we choose to take a planet from the inner solar system (Mercure)
# and one in the external solar system (Saturne)

# Initialization

G = 6.7e-11                                     # Gravitational constant

M_S = 5.7e26                                    # mass Saturne in kg
M_M = 3.3e23                                    # mass Mercure in kg
M_Sun = 2.0e30                                  # mass Sun in kg


# Normalization of ratings means adjustment of measured values
# at different scales at a theoretically common scale

M_n = 3.3e23                           # Normalized mass = masse Mercure

M_S = M_S/M_n                          # Normalized mass Sature
M_M = M_M/M_n                          # Normalized mass Mercure
M_Sun = M_Sun/M_n                      # Normalized mass Sun

N_d = 1.496e11      # Normalized distance in km, N_d = 1 astpos_S0nomical unit
N_t = 30*24*60*60     # Normalized time
G_G = (M_n*G*N_t**2)/(N_d**3)
t_init = 0                        # Initialized time = 0
t_final = 24         # Final time = 24 months = 2 years

Pts = 50*t_final                  # 50 points per month
t = np.linspace(t_init, t_final, Pts)
h = t[2] - t[1]                # Time step

# Positons vectors and velocities

pi_S = [1496e8/N_t, 0]         # Initial position Saturne
pi_M = [1496e8/N_t, 0]         # Initial position Mercure

pos_S = np.zeros([Pts, 2])     # Position vector Sature
v_S = np.zeros([Pts, 2])       # Velocity vector Sature

pos_M = np.zeros([Pts, 2])     # Position vector Mercure
v_M = np.zeros([Pts, 2])       # Velocity vector Mercure

K_E = np.zeros(Pts)           # Kinetic energy
P_E = np.zeros(Pts)           # Potential energy
A_M = np.zeros(Pts)           # Angular momentum
AreaVal = np.zeros(Pts)

Magn_S = np.sqrt(M_Sun*G_G/pi_S[0])  # Magnitude Saturne's initial velocity
Magn_M = 1.3e4 * N_t/N_d    # Magnitude Mercure's initial velocity
v_Si = [0, Magn_S*1]        # Initial velocity vector Saturne
v_Mi = [0, Magn_M*1]        # Initial velocity vector Mercure

# Initializing the arrays with the initial values

t_init = t[0]
pi_S = pos_S[0, :]
v_Si = v_S[0, :]
pi_M = pos_M[0, :]
v_Mi = v_M[0, :]


def init():

    """Initialization of trajectories for animation
    """
    line1.set_data([], [])
    line2.set_data([], [])
    txt.set_text('texte')

    return (line1, line2, txt)


def Force_S_Sun(pos_S):

    """Return the value of the attraction between Saturn and the Sun

    :param pos_S: Initial position vector of Saturne
    :type pos_S: ndarray of shape (n, 1)

    :return: Value of the attraction between Saturn and the Sun
    :rtype: float
    """

    F = np.zeros(2)
    Fmag = G_G*M_S*M_Sun/(np.linalg.norm(pos_S))**2
    theta = math.atan(np.abs(pos_S[1])/(np.abs(pos_S[0])))
    F[0] = Fmag * np.cos(theta)
    F[1] = Fmag * np.sin(theta)
    if pos_S[0] > 0:
        F[0] = -F[0]
    if pos_S[1] > 0:
        F[1] = -F[1]

    return F


def Force_M_Sun(pos_M):

    """Return the force of the attraction between Mercure and the Sun

    :param pos_M: Initial position vector of Mercure
    :type pos_M: ndarray of shape (n, 1)

    :return: Value of the attraction between Mercure and the Sun
    :rtype: float
    """

    F = np.zeros(2)
    Fmag = G_G*M_M*M_Sun/(np.linalg.norm(pos_S)+1e-20)**2
    theta = math.atan(np.abs(pos_S[1])/(np.abs(pos_S[0])+1e-20))
    F[0] = Fmag * np.cos(theta)
    F[1] = Fmag * np.sin(theta)
    if pos_S[0] > 0:
        F[0] = -F[0]
    if pos_S[1] > 0:
        F[1] = -F[1]

    return F


def Force_S_M(pos_Sa, pos_Me):

    """Return the force of the attraction between Saturn and Mercure

    :param pos_Sa: Position vector of Saturne
    :type pos_Sa: ndarray of shape (n, 1)

    :param pos_Me: Position vector of Mercure
    :type pos_Me: ndarray of shape (n, 1)

    :return: Value of the attraction between Saturn and Mercure
    :rtype: float
    """

    F = np.zeros(2)
    pos_S[0] = pos_Sa[0] - pos_Me[0]
    pos_S[1] = pos_Sa[1] - pos_Me[1]
    F_mag = G_G*M_S*M_M/(np.linalg.norm(pos_S))**2
    print(F_mag)
    theta = np.arctan(np.abs(pos_S[1])/(np.abs(pos_S[0])))
    F[0] = F_mag * np.cos(theta)
    F[1] = F_mag * np.sin(theta)
    if pos_S[0] > 0:
        F[0] = -F[0]
    if pos_S[1] > 0:
        F[1] = -F[1]

    return F


def Force(pos_S, star, pos_S0, v_S0):

    """ Returns the force between the planet and the Sun
    with the position and the initial velocity

    :param pos_S: Initial position vector of Saturne
    :type pos_S: ndarray of shape (n, 1)

    :param star:
    :type star: float

    :param pos_S0: Position vector of Saturne
    :type pos_S0: ndarray of shape (n, 1)

    :param v_S0: Velocity vector of Saturne
    :type v_S0: ndarray of shape (n, 1)

    :return: Force between the planet and the Sun
    """

    if star == 'Saturne':
        return Force_S_Sun(pos_S) + Force_S_M(pos_S, pos_S0)
    if star == 'Mercure':
        return Force_M_Sun(pos_S) - Force_S_M(pos_S, pos_S0)


def dpos_S_dt(t, pos_S, v_S, star, pos_S0, v_S0):
    return v_S


def dv_S_dt(t, pos_S, v_S, star, pos_S0, v_S0):

    F = Force(pos_S, star, pos_S0, v_S0)
    y = 0
    if star == 'Saturne':
        y = F/M_S
    if star == 'Mercure':
        y = F/M_M

    return y

# Differential Equation


def Euler(t, pos_S, v_S, h):

    """ Return a vector

    :param t: time
    :type t: float

    :param pos_S: Initial position vector of Saturne
    :type pos_S: ndarray of shape (n, 1)

    :param V_S: Initial position vector of Saturne
    :type V_S: Initial velocity vector of Saturne

    :param h: step time
    :type h: float

    :rtype: Vector
    """

    z = np.zeros([2, 2])
    pos_S1 = pos_S + h*dpos_S_dt(t, pos_S, v_S)
    v_S1 = v_S + h*dv_S_dt(t, pos_S, v_S)
    z = [pos_S1, v_S1]

    return z


def Euler_Crome(t, pos_S, v_S, h):

    z = np.zeros([2, 2])
    pos_S = pos_S + h*dpos_S_dt(t, pos_S, v_S)
    v_S = v_S + h*dv_S_dt(t, pos_S, v_S)
    z = [pos_S, v_S]

    return z


def Runge_Kutta_4(t, pos_S, v_S, h, star, pos_S0, v_S0):

    RK11 = dpos_S_dt(t, pos_S, v_S, star, pos_S0, v_S0)
    RK21 = dv_S_dt(t, pos_S, v_S, star, pos_S0, v_S0)

    RK12 = dpos_S_dt(t+0.5*h, pos_S+0.5*h*RK11,
                     v_S+0.5*h*RK21, star, pos_S0, v_S0)
    RK22 = dv_S_dt(t+0.5*h, pos_S+0.5*h*RK11, v_S+0.5*h*RK21,
                   star, pos_S0, v_S0)

    RK13 = dpos_S_dt(t+0.5*h, pos_S+0.5*h*RK12,
                     v_S+0.5*h*RK22, star, pos_S0, v_S0)
    RK23 = dv_S_dt(t+0.5*h, pos_S+0.5*h*RK12,
                   v_S+0.5*h*RK22, star, pos_S0, v_S0)

    RK14 = dpos_S_dt(t+h, pos_S+h*RK13, v_S+h*RK23, star, pos_S0, v_S0)
    RK24 = dv_S_dt(t+h, pos_S+h*RK13, v_S + h*RK23, star, pos_S0, v_S0)

    y0 = pos_S + h * (RK11+2.*RK12+2.*RK13+RK14) / 6.
    y1 = v_S + h * (RK21+2.*RK22+2.*RK23+RK24) / 6.

    z = np.zeros([2, 2])
    z = [y0, y1]

    return z


def Kinetic_Energy(v_S):

    v_N = np.linalg.norm(v_S)
    return 0.5*M_S*v_N**2


def Potential_Energy(pos_S):

    F_mag = np.linalg.norm(Force_S_Sun(pos_S))
    P_mag = np.linalg.norm(pos_S)

    return -F_mag*P_mag


def Kinetic_Moment(pos_S, v_S):

    r_N = np.linalg.norm(pos_S)
    v_N = np.linalg.norm(v_S)
    pos_S = pos_S/r_N
    v_S = v_S/v_N
    pos_S_dot_v_S = pos_S[0] * v_S[0] + pos_S[1] * v_S[1]
    theta = math.acos(pos_S_dot_v_S)

    return M_S*r_N*v_N*np.sin(theta)


def AreaCalc(pos_S1, pos_S2):

    pos_S1 = np.linalg.norm(pos_S1)
    pos_S2 = np.linalg.norm(pos_S2)
    pos_S1 = pos_S1 + 1e-20
    pos_S2 = pos_S2 + 1e-20
    theta_1 = math.atan(abs(pos_S1[1]/pos_S1[0]))
    theta_2 = math.atan(abs(pos_S2[1]/pos_S2[0]))
    pos_S = (pos_S1 + pos_S2) / 2.
    del_theta = np.abs(theta_1 - theta_2)

    return (del_theta*pos_S**2)/2.


def plot(fig, x, y, x_l, y_l, clr, lbl):

    py.figure(fig)
    py.xlabel(x_l)
    py.ylabel(y_l)

    return py.plot(x, y, clr, linewidth=1.0, label=lbl)


K_E[0] = Kinetic_Energy(v_S[0, :])
P_E[0] = Potential_Energy(pos_S[0, :])
A_M[0] = Kinetic_Moment(pos_S[0, :], v_S[0, :])
AreaVal[0] = 0


for i in range(0, Pts-1):

    [pos_S[i+1, :], v_S[i+1, :]] = Runge_Kutta_4(t[i], pos_S[i, :],
                                                 v_S[i, :], h, 'Saturne',
                                                 pos_M[i, :], v_M[i, :])
    [pos_M[i+1, :], v_M[i+1, :]] = Runge_Kutta_4(t[i], pos_M[i, :], v_M[i, :],
                                                 h, 'Mercure', pos_S[i, :],
                                                 v_S[i, :])

    K_E[i+1] = Kinetic_Energy(v_S[i+1, :])
    P_E[i+1] = Potential_Energy(pos_S[i+1, :])
    A_M[i+1] = Kinetic_Moment(pos_S[i+1, :], v_S[i+1, :])
    AreaVal[i+1] = AreaVal[i] + AreaCalc(pos_S[i, :], pos_S[i+1, :])


U_F = (G*M_n**2)/2
U_E = U_F * N_d

lbl = 'Orbit'
py.plot(0, 0, 'pos_S0', linewidth=7)
plot(1, pos_S[:, 0], pos_S[:, 1], r'$x$ position (AU)',
     r'$y$ position (AU)', 'green', 'Saturne')
plot(1, pos_M[:, 0], pos_M[:, 1], r'$x$ position (AU)',
     r'$y$ position (AU)', 'silver', 'Mercure')
py.ylim([-10, 10])

py.axis('Equal')
plot(2, t, K_E, r'Time, $t$ (Month)', r'Kinetic Energy, $K_E$ ($\times$'+str("% .*e"%(2, U_E))+'Joule', 'blue', 'K_E')
plot(2, t, P_E, r'Time, $t$ (Month)', r'Potential Energy, $K_E$ ($\times$'+str("%.*e"%(2, U_E))+'Joule', 'red', 'P_E')
plot(2, t, K_E+P_E, r'Time, $t$ (Month)', r'Total Energy, $K_E$ ($\times$'+str("%.*e"%(2, U_E))+'Joule', 'black', 'Total Energy')
q = py.legend(loc=0)
q.draw_frame(False)
py.ylim([-200, 200])

plot(3, t, A_M, r'Time, $t$ (Month)', r'Kinetic Moment', 'black', lbl)
py.ylim([2, 10])

plot(4, t, AreaVal, r'Time, $t$ (Month)',
     r'Sweeped Area ($AU^2$)', 'black', lbl)

########################
# Animation

def animate(i):


    Saturne_track = 40
    Mercure_track = 200
    time_month = 'Time elapsed = ' + str(round(t[i], 1)) + 'Month'
    txt.set_text(time_month)
    line1.set_data(pos_S[i: max(1, i-Saturne_track):-1, 0],
                   pos_S[i: max(1, i-Saturne_track):-1, 1])
    line2.set_data(pos_M[i: max(1, i-Mercure_track):-1, 0],
                   pos_M[i: max(1, i-Mercure_track):-1, 1])

    return (line1, line2)

##########################
# Function for animation

# fig, ax = py.subplots()
# ax.axis('square')
# ax.set_xlim((-8, 8))
# ax.set_ylim((-8, 8))
# ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

ax.plot(0, 0, 'o', markersize=9, markerfacecolor="#FDB813",
        markeredgecolor="#FD7813")
line1, = ax.plot([], [], 'o-', color = '#d2eeff', markevery=10000,
                markerfacecolor='#0077BE', linewidth=2)   # line for Saturne
line2, = ax.plot([], [], 'o-', color = '#e3dccb', markersize=8,
                markerfacecolor='#f66338', linewidth=2,
                markevery=10000)   # line for Mercure


ax.plot([-6,-5], [6.5,6.5], 'pos_S-')
ax.text(-4.5, 6.3, r'1 AU = $1.496 \times 10^8$ km')

ax.plot(-6, -6.2, 'o', color='#d2eeff', markerfacecolor='#0077BE')
ax.text(-5.5, -6.4, 'Saturne')

ax.plot(-3.3, -6.2, 'o', color='#e3dccb',
        markersize=8, markerfacecolor='#f66338')
ax.text(-2.9, -6.4, 'Mercure')

ax.plot(5, -6.2, 'o', markersize=9, markerfacecolor="#FDB813",
        markeredgecolor="#FD7813")
ax.text(5.5, -6.4, 'Sun')

txt = ax.text(0.24, 1.05, '--', transform=ax.transAxes, va='center')
plt.title('Time elapsed, T=%i Month' %u)

###############################
# Call animation function

animation = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=4000, interval=5, blit=True)

HTML(animation.to_html5_video())

# Enable the following line if you want to save the animation to file.
anim.save('Orbit.mp4', fps=30, dpi=500, extra_args=['-vcodec', 'libx264'])
