import numpy as np
import matplotlib.pyplot as plt

D = 0.1
D_b = 0.05
A_cont = 1*1e-4
A_s = 4*np.pi*(D/2)**2 - A_cont
Q_fc = 1
mu_tot = 0.02
rho_c_p_b = 3*1e+6
epsilon_s = 0.01
x = 0.0025
k_in = 0.005
x_sw = 0.03
k_sw = 59
A_sw = 1.5*1e-6
epsilon_s_vec = 0.09
A_vec = 0.75*A_s

g = 3.71
U = 2
r = 206.6 - 249.2*1e+9
T_sun = 5780
r_s = 0.696*1e+9
sigma = 5.67*1e-8

T_inf = 50
wind = True


def get_parameters(T_inf):
    T_inf_array = np.arange(50, 300, 50)
    index = np.argmin(np.abs(T_inf_array - T_inf))

    k = [0.0037, 0.0057, 0.0076, 0.0094, 0.0112]
    C_p = [670.5545, 715.3841, 758.1273, 798.8477, 837.6091]
    rho = [0.0794, 0.0397, 0.0265, 0.0198, 0.0159]

    alpha = [6.979*1e-5, 2.0225*1e-4, 3.7907*1e-4, 5.935*1e-4, 8.4133*1e-4]
    visc = [9.7172*1e-5, 2.7484*1e-4, 5.0492*1e-4, 7.7737*1e-4, 11*1e-4]
    P_r = [1.3923, 1.3589, 1.332, 1.3098, 1.2913]

    return k[index], C_p[index], rho[index],\
        alpha[index], visc[index], P_r[index]

def get_Nu_from_R_e(R_e, P_r):
    Nu = None

    if R_e > 0.4 and R_e <= 4:
        Nu = 0.989*R_e**0.33*P_r**(1/3)
    elif R_e > 4 and R_e <= 40:
        Nu = 0.911*R_e**0.385*P_r**(1/3)
    elif R_e > 40 and R_e <= 4000:
        Nu = 0.683*R_e**0.466*P_r**(1/3)
    elif R_e > 4000 and R_e <= 40000:
        Nu = 0.193*R_e**0.618*P_r**(1/3)
    elif R_e > 40000 and R_e <= 400000:
        Nu = 0.027*R_e**0.805*P_r**(1/3)

    return Nu

def compute_dQ_loss(T_inf, T_b, wind, show):
    k, C_p, rho, alpha, visc, P_r = get_parameters(T_inf)

    # Q_cond = k*A*(T_1 - T_2)/L
    # Q_conv = h*A*(T_s - T_inf)
    # Q_rad = epsilon*sigma*A_s*(T_s**4 - T_surr**4)
    # beta = 1/T_inf
    # h = nu*k/L
    # h_r = sigma*epsilon*(T_s + T_inf)*(T_s**2 + T_inf**2)

    R_cond_sw = x_sw/(k_sw*A_sw)
    R_cond_g = x/(k_in*A_cont)

    T_s = T_b

    for i in range(10):
        R_cond_s = x/(k_in*A_s)

        if wind:
            R_e = U*D/visc
            Nu = get_Nu_from_R_e(R_e, P_r)
        else:
            beta = 1/T_inf
            R_a = (g*beta*(np.abs(T_s - T_inf))*D**3)/(visc**2) # Possibilite d'etre imaginaire??
            Nu = 2 + (0.589*R_a**(1/4))/((1 + (0.469/P_r)**(9/16))**(4/9))

        h = Nu*k/D
        R_conv = 1/(h*A_s)

        h_r_vec = sigma*epsilon_s_vec*(T_s + T_inf)*(T_s**2 + T_inf**2)
        R_rad_vec = 1/(h_r_vec*A_vec)

        h_r_s = sigma*epsilon_s*(T_s + T_inf)*(T_s**2 + T_inf**2)
        R_rad = 1/(h_r_s*(A_s - A_vec))

        R_eq_ext = 1/(1/R_conv + 1/R_rad_vec + 1/R_rad)
        T_s = (T_b - T_inf)*(R_eq_ext/(R_cond_s + R_eq_ext))

    R_eq = 1/(1/R_cond_sw + 1/R_cond_g + 1/(R_cond_s + R_eq_ext))

    dQ_loss = (T_b - T_inf)/R_eq

    if show:
        print("R_cond_sw: " + str(R_cond_sw))
        print("R_cond_g: " + str(R_cond_g))
        print("R_cond_s: " + str(R_cond_s))
        print("R_conv: " + str(R_conv))
        print("R_rad_vec: " + str(R_rad_vec))
        print("R_rad: " + str(R_rad))
        print("R_eq_ext: " + str(R_eq_ext))
        print("T_s: " + str(T_s))
        print("dQ_loss: " + str(dQ_loss))

    return dQ_loss

def compute_dQ_tot(T_inf, T_b, wind, show):
    dQ_sun = sigma*epsilon_s*(A_s/2)*(r_s/r)**2*T_sun**4
    dQ_int = (1-mu_tot)*Q_fc
    dQ_loss = compute_dQ_loss(T_inf, T_b, wind, show)
    dQ_tot = dQ_int + dQ_sun - dQ_loss
    return dQ_tot


compute_dQ_loss(T_inf, 300, False, True)
#compute_dQ_loss(250, False) #Hottest
#compute_dQ_loss(50, True) #Coldest

time = 3600
time_array = np.arange(0, time, 1)
T_b = 300
Q_tot = 0
Q_tot = np.zeros(time)
T_b_array = np.zeros([2, time])
T_b_array = T_b_array + 300
volume = (4*np.pi*(D_b/2)**3)/3

for i in range(2):
    for j in range(1, time):
        dQ_tot = None
        if i == 0:
            dQ_tot = compute_dQ_tot(50, T_b_array[i][j], True, False)
        elif i == 1:
            dQ_tot = compute_dQ_tot(250, T_b_array[i][j], False, False)

        Q_tot[j] = Q_tot[j-1] + dQ_tot
        T_b_array[i][j] = T_b_array[i][j-1] + Q_tot[j]/(volume*rho_c_p_b) #Potentiellement ajouter le gaz a cela?


fig, ax = plt.subplots()

T_inf_array = np.array([50, 250])
wind_array = np.array([True, False])
for i in range(len(T_inf_array)):
    ax.plot(time_array, T_b_array[i], label="T_inf: " + str(T_inf_array[i]) + " K and Wind: " + str(wind_array[i]))
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Temperature du robot en fonction du temps")

ax.legend()
ax.grid(True)
plt.show()
