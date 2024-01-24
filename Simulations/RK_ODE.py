import numpy as np


def RK4(f, t0, tf, y0, nbPoints, args_f, noise, args_b):
    """"
    Résout l'EDO décrite par l'équation Y' = f(Y) avec la méthode RK4 et renvoit le tableau de temps et de solution avec:
    -t0 : temps initial
    -tf : temps final
    -y0 : valeur initiale
    -nbPoints : nombre de points
    -args_f : les arguments de la fonction f
    -noise : la fonction de bruitage
    -args_b: les arguments de la fonction de bruitage

    return : -t numpy array tableaux des temps
             -y numpy array de taille (temps*y0) contenant la solution
    """
    t = np.linspace(t0, tf, nbPoints)
    dt = t[1] - t[0]
    y = np.zeros((t.size,) + y0.shape)
    y[0] = y0
    for i in range(t.shape[0] - 1):
        p1 = f(t[i], y[i], dt, args_f)
        p2 = f(t[i] + 0.5 * dt, y[i] + dt * 0.5 * p1, dt, args_f)
        p3 = f(t[i] + 0.5 * dt, y[i] + dt * 0.5 * p2, dt, args_f)
        p4 = f(t[i] + 1 * dt, y[i] + dt * p3, dt, args_f)

        y[i + 1] = noise(y[i] + (1 / 6) * dt * (p1 + 2 * p2 + 2 * p3 + p4), dt, args_b)

    return t, y


def bruit(y, dt, argb):
    """"
    Introduction un bruit angulaire à chaque étape en tournant n et v:
    y: le tableau a faire tourner
    dt: le pas de temps
    argb: l'applitssement de la gaussienne de bruit
    """
    var = 2 * argb * dt
    dtheta = np.random.normal(0, var)

    y[2], y[3] = 0, -y[2] * np.sin(dtheta) + y[3] * np.cos(dtheta)
    y[4] = y[4] + dtheta

    return y


def step(t, y, dt, args):
    """
    Permet de passer de la solution à un temps ti à la solution au temps ti + dt
    t: tableau de temps
    y: la solution au temps ti
    dt: pas de temps
    args: les arguments du problème
    """

    rx, ry, vx, vy, theta, omega = y

    tau_v, kappa, sigma, tau_n, beta, I, lambd, tau_r, N_w = args

    alpha = 20  #Force du mur

    #Définition des vecteurs du problème (en 3D)
    n = np.array([np.cos(theta), np.sin(theta), 0])
    v = np.array([vx, vy, 0])
    f = - kappa * np.array([np.power(sigma, -alpha) * np.power(rx, alpha-1)*(rx >= 0),
                            0, 0])

    lambd = lambd * (rx >= 4.575)
    tau_r = tau_r * (rx >= 4.575)

    #Derivées obtenues à partir des équations
    dvx = 0# (np.cos(theta) - vx - N_w * (rx >= 4.575) + f[0]) * (1.0 / tau_v)
    dvy = (np.sin(theta) - (1 + lambd) * vy - omega * lambd/2) * (1.0 / tau_v)

    domega = (1.0 / I) * ((np.cos(theta) - tau_r)*vy - np.sin(theta)*vx - (tau_n + tau_r/2)*omega + beta)

    return np.array([vx,
                     vy,
                     dvx,
                     dvy,
                     omega,
                     domega])


def ode_solution(my_arg):
    """"
    Prend un entrée les arguments du problèmes et renvoi le tableau de temps et la solution
    sous la forme [x, y, vx, vy, theta, omega]
    """
    x0, y0, vx0, vy0, theta0, omega_0, tau_v, kappa, sigma, tau_n, beta, I, lambd, tau_r, N_w, t_max, nb_points, D = my_arg

    nb_points = int(nb_points)

    condinit = np.array([x0, y0, vx0, vy0, theta0, omega_0])

    their_args = [tau_v, kappa, sigma, tau_n, beta, I, lambd, tau_r, N_w]

    t, sol = RK4(step, 0, t_max, condinit, nb_points, their_args, bruit, D)

    return t, sol
