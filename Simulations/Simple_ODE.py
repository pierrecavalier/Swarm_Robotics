import numpy as np
from scipy.integrate import odeint
from numpy.fft import rfftfreq, rfft
from matplotlib import cm
from matplotlib.ticker import LinearLocator

#Ancien fichier de résolution du l'équation différentielle couplée: fonctionne avec odeint et n'est pas tenu à jour

def force_potentiel(r,sigma,alpha):
    # -gradient du potentiel : (rx/sigma)**alpha + (ry/sigma)**alpha
    return -(alpha*(r)**(alpha-1))/(sigma**(alpha))

def ode_step(y, t, g, tau_v, tau_n, bias,sigma,alpha):
    rx, ry, vx, vy, theta, *rest = y
    particles = [(rx, ry, vx, vy, theta)]

    while len(rest) > 0:
        rx, ry, vx, vy, theta, *rest = rest
        particles.append((rx, ry, vx, vy, theta))\

    dydt = list()
    for y in particles:
        rx, ry, vx, vy, theta = y
        forces_x = force_potentiel(rx,sigma,alpha) + g
        forces_y = force_potentiel(ry,sigma,alpha)

        dydt.append(vx)
        dydt.append(vy)
        dydt.append((np.cos(theta) - vx + forces_x) * (1.0/tau_v))
        dydt.append((np.sin(theta) - vy + forces_y) * (1.0/tau_v))
        dydt.append((vy*np.cos(theta) - vx*np.sin(theta) + bias) * (1.0/tau_n))

    return dydt


def ode_solution(x0, y0, vx0, vy0, theta0, g, tau_v, tau_n, bias,sigma,alpha, t_max, nb_points):
    cond_init = list()
    cond_init.append(x0)
    cond_init.append(y0)
    cond_init.append(vx0)
    cond_init.append(vy0)
    cond_init.append(theta0)

    t = np.linspace(0, t_max, nb_points)
    sol = odeint(ode_step, cond_init, t, args=(g, tau_v, tau_n, bias,sigma,alpha))

    return sol


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lg = [0.5, 1, 1.5]
    ltau_n = [0.5+0.01 * i for i in range(1, 21)]
    lbias = [0.5+0.01 * i for i in range(1, 21)]

    for g in lg:
        res = np.zeros((len(ltau_n), len(lbias)))
        # plt.title(f"Number of direction loops - max angle /2pi- capped to 4\n vx0=0, vy0=1,, tau_v=10e-6, "
        #           f"t_max=50, dt=0.25, g={g}")
        for i in range(len(ltau_n)):
            for j in range(len(lbias)):
                n = 20000
                dt = 0.01
                sol = ode_solution(0, 0, 0, 1, np.pi/2, g, 10e-6, ltau_n[i], lbias[j], round(n*dt), n)

                data = sol[:, 4] % (2*np.pi)
                data -= data.mean()
                fourier = rfft(data)
                freq = rfftfreq(n, d=dt)
                res[i, j] = 1.0 / freq[np.absolute(fourier).argmax()]

                # print(f"Period = {1.0/freq[np.absolute(fourier).argmax()]}, b={lbias[j]}, taun={ltau_n[i]}")

                # plt.title(f"Period = {1.0/freq[np.absolute(fourier).argmax()]}, b={lbias[j]}, taun={ltau_n[i]}")
                # plt.semilogy(freq, np.absolute(fourier), label="abs")
                # # plt.plot(freq, fourier.imag, label="imag")
                # plt.legend()
                # plt.show()

                # for k in range(50, len(fourier)):
                #     fourier[k] = 0
                #
                # plt.plot(np.linspace(0, n*dt, num=n), data)
                # plt.plot(np.linspace(0, n*dt, num=n), np.fft.irfft(fourier))
                # plt.show()
        # print(res)

        # X = np.array(ltau_n)
        # Y = np.array(lbias)
        # X, Y = np.meshgrid(X, Y)
        #
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # # Plot the surface.
        # surf = ax.plot_surface(X, Y, res.T,
        #                        linewidth=0, antialiased=True, vmin=0, vmax=25)
        #
        # # Customize the z axis.
        # # ax.set_zlim(-1.01, 1.01)
        # # ax.zaxis.set_major_locator(LinearLocator(10))
        # # # A StrMethodFormatter is used automatically
        # # ax.zaxis.set_major_formatter('{x:.02f}')
        #
        # # Add a color bar which maps values to colors.
        # # fig.colorbar(surf, ax=ax)
        plt.xlabel("taun")
        plt.ylabel("b")
        # plt.title(f"Loop period, g={g}, t_max=50, dt=0.01")
        # plt.show()

        plt.title(f"Loop period, g={g}, t_max=200, dt=0.01")
        plt.imshow(res.T, cmap="coolwarm", origin="lower", extent=(ltau_n[0], ltau_n[-1], lbias[0], lbias[-1]))
        plt.colorbar()
        plt.show()


