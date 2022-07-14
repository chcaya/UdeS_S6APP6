import numpy as np


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
U = np.arange(0, 2.1, 0.1)
r = 206.6 - 249.2*1e+9
T_sun = 5780
r_s = 0.696*1e+9
sigma = 5.67*1e-8

T_inf = np.arange(50, 300, 50)
k = [0.0037, 0.0057, 0.0076, 0.0094, 0.0112]
C_p = [670.5545, 715.3841, 758.1273, 798.8477, 837.6091]
rho = [0.0794, 0.0397, 0.0265, 0.0198, 0.0159]

alpha = [6.979*1e-5, 2.0225*1e-4, 3.7907*1e-4, 5.935*1e-4, 8.4133*1e-4]
visc = [9.7172*1e-5, 2.7484*1e-4, 5.0492*1e-4, 7.7737*1e-4, 11*1e-4]
P_r = [1.3923, 1.3589, 1.332, 1.3098, 1.2913]

scenario = 0

T_inf = T_inf[scenario]
k = k[scenario]
C_p = C_p[scenario]
rho = rho[scenario]

alpha = alpha[scenario]
visc = visc[scenario]
P_r = P_r[scenario]

T_b = 300

# Q_cond = k*A*(T_1 - T_2)/L
# Q_conv = h*A*(T_s - T_inf)
# Q_rad = epsilon*sigma*A_s*(T_s**4 - T_surr**4)
# beta = 1/T_inf
# h = nu*k/L
# h_r = sigma*epsilon*(T_s + T_inf)*(T_s**2 + T_inf**2)

Q_sun = sigma*epsilon_s*(A_s/2)*(r_s/r)**2*T_sun**4
Q_int = (1-mu_tot)*Q_fc

R_cond_sw = x_sw/(k_sw*A_sw)
R_cond_g = x/(k_in*A_cont)

T_s = T_b

for i in range(10):
    R_cond_s = x/(k_in*A_s)

    beta = 1/T_inf
    R_a = (g*beta*(T_s - T_inf)*D**3)/(visc**2)
    nu = 2 + (0.589*R_a**(1/4))/((1 + (0.469/P_r)**(9/16))**(4/9))
    h = nu*k/D
    R_conv = 1/(h*A_s)

    h_r_vec = sigma*epsilon_s_vec*(T_s + T_inf)*(T_s**2 + T_inf**2)
    R_rad_vec = 1/(h_r_vec*A_vec)

    h_r_s = sigma*epsilon_s*(T_s + T_inf)*(T_s**2 + T_inf**2)
    R_rad = 1/(h_r_s*(A_s - A_vec))

    R_eq_ext = 1/(1/R_conv + 1/R_rad_vec + 1/R_rad)
    T_s = (T_b - T_inf)*(R_eq_ext/(R_cond_s + R_eq_ext))

R_eq = 1/(1/R_cond_sw + 1/R_cond_g + 1/(R_cond_s + R_eq_ext))

Q_eq = (T_b - T_inf)/R_eq

Q_tot = Q_int + Q_sun + Q_eq

print("R_cond_sw: " + str(R_cond_sw))
print("R_cond_g: " + str(R_cond_g))
print("R_cond_s: " + str(R_cond_s))
print("R_conv: " + str(R_conv))
print("R_rad_vec: " + str(R_rad_vec))
print("R_rad: " + str(R_rad))
print("R_eq_ext: " + str(R_eq_ext))
print("T_s: " + str(T_s))
print("Q_tot: " + str(Q_tot))
