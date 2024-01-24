from RK_ODE import ode_solution
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


x0 = 5
y0 = 0
vx0 = 0
vy0 = 0
omega_0 = 0
lambd = 0.10
tau_v = 10
kappa = 10
sigma = float(5)
tau_n = 0.01
beta = 0
I = 1.5
tau_r = 1.5
N_w = 0

D = 0
nb_points = 1000
t_max = 200

#0 tau_v varie pour theta fixé
#1 theta varie pour plusieurs lambda

mode = 2

if mode == 0:  #Tau v varie pour theta fixé
    tauv_tab = np.linspace(1, 15, 8)
    theta0_tab = [0.4]

    for tau_v in tauv_tab:
        for theta0 in theta0_tab:
            my_arg = x0, y0, vx0, vy0, theta0, omega_0, tau_v, kappa, sigma, tau_n, beta, I, lambd, tau_r, N_w, t_max, nb_points, D

            t, sol = ode_solution(my_arg)

            rx = sol[:, 1]
            theta = sol[:, 4]

            plt.plot(theta, label=f" $\u03C4_v$ = {tau_v}")

    plt.title(f" $\tau_v$ = {tau_v}")

    anglelim = 1
    lim = np.array([anglelim, anglelim])
    plt.plot([0, nb_points], lim, color="black")
    plt.plot([0, nb_points], -lim, color="black")
    plt.show()

if mode == 1:  #Plot different theta pour different lambda puis angle lim
    lamba_tab = np.linspace(0.01, 1, 100)
    theta0_tab = np.linspace(0.2, 0.6, 100)

    tab_anglelim = []

    for lambd in lamba_tab:
        toadd = 0
        for theta0 in theta0_tab:
            my_arg = x0, y0, vx0, vy0, theta0, omega_0, tau_v, kappa, sigma, tau_n, beta, I, lambd, tau_r, N_w, t_max, nb_points, D

            t, sol = ode_solution(my_arg)

            rx = sol[:, 1]
            theta = sol[:, 4]

            plt.plot(theta)

            if len(theta[np.abs(theta) < 1.5]) == len(theta):
                toadd = theta0

        plt.title(f" $\lambda$ = {np.round(lambd, 2)}")

        anglelim = 1
        lim = np.array([anglelim, anglelim])
        plt.plot([0, nb_points], lim, color="black")
        plt.plot([0, nb_points], -lim, color="black")
        plt.show()

        tab_anglelim.append(toadd)

    print(tab_anglelim)

    fig = plt.figure()
    plt.plot(lamba_tab, tab_anglelim)
    plt.title("Angle limite en fonction de $\lambda$")
    plt.show()

    print("Lambda: ", np.array(lamba_tab), "Angle lim: ", np.array(tab_anglelim))

if mode == 2:
    x = np.linspace(0.1, 1, 100)

    y = [0.23232323, 0.24040404, 0.24848485, 0.26060606, 0.26868687, 0.27676768,
         0.28484848, 0.29292929, 0.30505051, 0.31313131, 0.32121212, 0.32929293,
         0.33737374, 0.34545455, 0.35353535, 0.36161616, 0.36969697, 0.37777778,
         0.38585859, 0.38989899, 0.3979798,  0.40606061, 0.41414141, 0.41818182,
         0.42626263, 0.43434343, 0.43838384, 0.44646465, 0.45050505, 0.45858586,
         0.46262626, 0.47070707, 0.47474747, 0.48282828, 0.48686869, 0.49090909,
         0.49090909, 0.49494949, 0.4989899,  0.4989899, 0.5030303,  0.5030303,
         0.50707071, 0.51111111, 0.51111111, 0.51111111, 0.51515152, 0.51919192,
         0.51919192, 0.51919192, 0.51919192, 0.51919192, 0.52323232, 0.52727273,
         0.52727273, 0.53131313, 0.53131313, 0.53131313, 0.53131313, 0.53535354,
         0.53535354, 0.53939394, 0.54343434, 0.54747475, 0.54747475, 0.54747475,
         0.54747475, 0.54747475, 0.54747475, 0.54747475, 0.54747475, 0.54747475,
         0.55151515, 0.55151515, 0.55151515, 0.55555556, 0.55959596, 0.55959596,
         0.55959596, 0.55959596, 0.56363636, 0.56363636, 0.56363636, 0.56363636,
         0.56363636, 0.56363636, 0.56363636, 0.56363636, 0.56363636, 0.56767677,
         0.56767677, 0.56767677, 0.56767677, 0.56767677, 0.57171717, 0.57575758,
         0.57575758, 0.57575758, 0.57575758, 0.57575758]


    def log(data, a, b, c):
        return a * np.log(data + b) + c

    def racine(data, a, b, c, alpha):
        return a * np.power((data + b), alpha) + c

    x = x[30:]
    y = y[30:]

    param0 = np.array([1, 1, 0])
    p_opt, cov = curve_fit(log, x, y, param0, bounds=([-10, 0, -10], [10, 10, 10]))

    print("parametres : ", p_opt)
    print("matrice de covariance : ", cov)
    a_opt = p_opt[0]
    b_opt = p_opt[1]
    c_opt = p_opt[2]

    ybis = log(x, a_opt, b_opt, c_opt)
    plt.plot(x, log(x, a_opt, b_opt, c_opt), label=f"Modele log")

    param0 = np.array([1, 1, 0, 1])
    p_opt, cov = curve_fit(racine, x, y, param0, bounds=([-10, 0, -10, -10], [10, 10, 10, 10]))

    print("parametres : ", p_opt)
    print("matrice de covariance :", cov)

    a_opt = p_opt[0]
    b_opt = p_opt[1]
    c_opt = p_opt[2]
    alpha_opt = p_opt[3]

    plt.plot(x, racine(x, a_opt, b_opt, c_opt, alpha_opt), label=f"Modele x^alpha")

    plt.scatter(x, y, label="Simulation")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\theta$ décrochage")
    plt.legend()
    plt.show()
