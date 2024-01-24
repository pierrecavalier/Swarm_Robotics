from RK_ODE import ode_solution
import matplotlib.pyplot as plt
import numpy as np


x0 = 0
y0 = 0
vx0 = 0
vy0 = 0
theta0 = 0
omega0 = [0.1,0.2,-0.05,-0.2]

tauv = 1
kappa = 0
lamb = 0.1
taun = [0.1,0.3,0.5,1,1.5,2,5]
beta = 0
eta = np.linspace(-0,-10,200)

tmax = 300
nb = 3000


for tau_n in taun:
    res = np.zeros((len(eta), len(omega0),))
    for j in range(len(omega0)):
        for i in range(len(eta)):
            arg = [x0, y0, vx0, vy0, theta0, omega0[j], tauv, kappa, lamb, tau_n, beta, eta[i], tmax, nb, 0]
            x,y,vx,vy,theta,omega = ode_solution(arg)[1][-1]

            phi = np.arctan((vy/vx))
            res[i,j] = phi -theta
            print(i)

    fig = plt.figure()
    plt.plot(eta,res)
    plt.title(tau_n)
    plt.savefig(str(tau_n)+".png")





