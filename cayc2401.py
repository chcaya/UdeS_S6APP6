import numpy as np

D = 0.1
D_b = 0.05
A_cont = 1*1e-4
A_s = 4*np.pi*(D/2)**2 - Acont
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
U = np.pi

# Q_cond = k*A*(T_1 - T_2)/L
# Q_conv = h*A*(T_1 - T_2)
# Q_rad = epsilon*sigma*A_s*(T_s**4 - T_surr**4)
