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
x = 0.025
k_in = 0.005
x_sw = 0.03
k_sw = 59
A_sw = 1.5*1e-6
epsilon_s_vec = 0.09
A_vec = 0.75*A_s

g = 3.71
U = 2
T_sun = 5780
r_s = 0.696*1e+9
sigma = 5.67*1e-8

dQ_int = (1-mu_tot)*Q_fc


def get_parameters(T_inf):
    T_inf_array = np.arange(50, 300, 50)
    index = np.argmin(np.abs(T_inf_array - T_inf))

    k = [0.0037, 0.0057, 0.0076, 0.0094, 0.0112]
    C_p = [670.5545, 715.3841, 758.1273, 798.8477, 837.6091]
    rho = [0.0794, 0.0397, 0.0265, 0.0198, 0.0159]

    alpha = [6.979*1e-5, 2.0225*1e-4, 3.7907*1e-4, 5.935*1e-4, 8.4133*1e-4]
    visc = [9.7172*1e-5, 2.7484*1e-4, 5.0492*1e-4, 7.7737*1e-4, 11*1e-4]
    P_r = [1.3923, 1.3589, 1.332, 1.3098, 1.2913]

    r = np.linspace(206.6*1e+9, 249.2*1e+9, len(k))
    r = np.flip(r)

    return k[index], C_p[index], rho[index],\
        alpha[index], visc[index], P_r[index], r[index]


def compute_dQ_loss(T_inf, T_b, wind, vec_on, sw_on, show):
    k, C_p, rho, alpha, visc, P_r, r = get_parameters(T_inf)

    r2 = D/2
    r1 = r2 - x
    R_cond_s = (r2 - r1)/(4*np.pi*r1*r2*k_in)

    T_s = T_b

    R_cond_sw = x_sw/(k_sw*A_sw)
    R_cond_g = x/(k_in*A_cont)

    for i in range(10):
        Nu = None
        R_e = None
        R_a = None
        if wind:
            R_e = U*D/visc
            mu_ratio = 1
            Nu = 2 + (0.4*R_e**(1/2) + 0.06*R_e**(2/3))*P_r**0.4*(mu_ratio)**(1/4)
        else:
            beta = 1/T_inf
            R_a = (P_r*g*beta*(np.abs(T_s - T_inf))*D**3)/(visc**2)
            Nu = 2 + (0.589*R_a**(1/4))/((1 + (0.469/P_r)**(9/16))**(4/9))

        h = Nu*k/D
        R_conv = 1/(h*A_s)

        h_r_vec = sigma*epsilon_s_vec*(T_s + T_inf)*(T_s**2 + T_inf**2)
        R_rad_vec = 1/(h_r_vec*A_vec)
            
        h_r_s = sigma*epsilon_s*(T_s + T_inf)*(T_s**2 + T_inf**2)
        R_rad = 1/(h_r_s*(A_s - A_vec))

        R_eq_ext = None
        if vec_on:
            R_eq_ext = 1/(1/R_conv + 1/R_rad_vec + 1/R_rad)
        else:
            R_eq_ext = 1/(1/R_conv + 1/R_rad)

        T_s = T_b - (T_b - T_inf)*(R_cond_s/(R_cond_s + R_eq_ext))

    R_eq = None
    if sw_on:
        R_eq = 1/(1/R_cond_g + 1/R_cond_sw + 1/(R_cond_s + R_eq_ext))
    else:
        R_eq = 1/(1/R_cond_g + 1/(R_cond_s + R_eq_ext))

    dQ_loss = (T_b - T_inf)/R_eq

    dQ_cond_s = (T_b - T_s)/R_cond_s
    dQ_cond_sw = (T_b - T_inf)/R_cond_sw
    dQ_cond_g = (T_b - T_inf)/R_cond_g
    dQ_conv = (T_s - T_inf)/R_conv
    dQ_rad_vec = (T_s - T_inf)/R_rad_vec
    dQ_rad = (T_s - T_inf)/R_rad

    if show:
        print("R_cond_s: " + str(R_cond_s))
        print("R_cond_sw: " + str(R_cond_sw))
        print("R_cond_g: " + str(R_cond_g))
        print("R_conv: " + str(R_conv))
        print("R_rad_vec: " + str(R_rad_vec))
        print("R_rad: " + str(R_rad))
        print("R_eq_ext: " + str(R_eq_ext))
        print("R_eq: " + str(R_eq))
        print("T_s: " + str(T_s))
        print("R_a: " + str(R_a))
        print("R_e: " + str(R_e))
        print("Nu: " + str(Nu))
        print("dQ_cond_s: " + str(dQ_cond_s))
        print("dQ_cond_sw: " + str(dQ_cond_sw))
        print("dQ_cond_g: " + str(dQ_cond_g))
        print("dQ_conv: " + str(dQ_conv))
        print("dQ_rad_vec: " + str(dQ_rad_vec))
        print("dQ_rad: " + str(dQ_rad))
        print("dQ_loss: " + str(dQ_loss))

    return dQ_loss


def compute_dQ_tot(T_inf, T_b, prev_dQ_tot, wind, vec_on, sw_on, show):
    k, C_p, rho, alpha, visc, P_r, r = get_parameters(T_inf)
    dQ_sun = sigma*epsilon_s*(A_s/2)*(r_s/r)**2*T_sun**4
    dQ_loss = compute_dQ_loss(T_inf, T_b, wind, vec_on, sw_on, False)
    dQ_tot = dQ_int + dQ_sun - dQ_loss

    if show:
        print("dQ_sun: " + str(dQ_sun))
        print("dQ_int: " + str(dQ_int))
        print("dQ_loss: " + str(dQ_loss))
        print("dQ_tot: " + str(dQ_tot))

    return dQ_tot


def compute_T_b():
    time = 3000
    dt = 60
    time_array = np.arange(0, time, 1)
    T_b = 300
    volume = (4*np.pi*(D_b/2)**3)/3
    Q_tot = np.zeros(time)
    Q_tot[0] = rho_c_p_b*volume*T_b
    T_b_array = np.zeros([2, time])
    T_b_array[:, 0] = T_b
    dQ_tot = 0

    for i in range(2):
        for j in range(1, time):
            if i == 0:
                dQ_tot = compute_dQ_tot(50, T_b_array[i][j-1], dQ_tot, True, True, True, False)
            elif i == 1:
                dQ_tot = compute_dQ_tot(250, T_b_array[i][j-1], dQ_tot, False, True, True, False)

            Q_tot[j] = Q_tot[j-1] + dQ_tot*dt
            T_b_array[i][j] = Q_tot[j]/(volume*rho_c_p_b) #Potentiellement ajouter le gaz a cela?

            if T_b_array[i][j] < 0:
                T_b_array[i][j] = 0

    return time_array, T_b_array


def plot_temperature(time_array, T_b_array):
    fig, ax = plt.subplots()

    T_inf_array = np.array([50, 250])
    wind_array = np.array([True, False])
    for i in range(len(T_inf_array)):
        ax.plot(time_array, T_b_array[i], label="T_inf: " + str(T_inf_array[i]) + "K and Wind: " + str(wind_array[i]))
        ax.set_xlabel("Temps (min)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title("Temperature du robot en fonction du temps")
        print("Temperature (" + str(T_inf_array[i]) + "K): " + str(T_b_array[i][-1]) + "K")

    ax.legend()
    ax.grid(True)
    plt.show()

print("-----Cas 1-----")
compute_dQ_loss(50, 300, False, True, True, True)
print("-----Cas 2-----")
compute_dQ_loss(50, 300, True, True, True, True)
print("-----Cas 3-----")
compute_dQ_loss(250, 300, False, True, True, True)
print("-----Cas 4-----")
compute_dQ_loss(250, 300, True, True, True, True)
print()

time_array, T_b_array = compute_T_b()
plot_temperature(time_array, T_b_array)
