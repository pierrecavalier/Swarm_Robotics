import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from RK_ODE import ode_solution


# Objectif de ce code : Trouver les paramètres qui conditionne les valeurs propres de la matrice ci dessous
# [[0, 1, 0]
# [0, -tau/J, (1-tau_r)/J]
# [1/tau_v, -lambd/(2*tau_v), - (1+lambd)/tau_v]] qui régit le système dans le cas limite theta petit et dynamique
# selon x négligée. Par ailleurs vp = valeurs propre /eigenvalues

def matrice(tau_r, tau_n, tau_v, J, lambd):
    """
    Renvoit le déterminant de la matrice du problème
    """
    tau = tau_n + tau_r / 2
    mat = [[0, 1, 0], [0, -tau / J, (1 - tau_r) / J], [1 / tau_v, -lambd / (2 * tau_v), - (1 + lambd) / tau_v]]
    return np.linalg.eigvals(mat)


def decimer(liste, r, prob):
    """
    Permet d'obtenir une distribution de points distribués de facon homogène:
    à appeller après avoir générer une liste de jeux de parametre valide avec Monte-Carlo

    liste : numpy array  ou liste contenant une liste pour la coorrdonnée x et une pour y
    r : float taille du disque utilisé pour caluler la densite locale de point
    prob : float entre 0 et 1 probabilité pour les points de plus haute densité de se faire supprimer

    return : tuple coordonnées des points conservés
    """

    tab = np.array(liste)
    x, y = tab[0, :], tab[1, :]
    print(len(x), len(y))
    densite = np.zeros(len(x))

    for i in range(len(x)):
        dist = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
        densite[i] = np.count_nonzero(dist <= r)
        if i % 100 == 0:
            print(i)

    rho_min = np.min(densite)
    rho_max = np.max(densite)
    print(rho_min, rho_max)
    d = prob * (1 - (rho_min / rho_max)) / (rho_max - rho_min)
    proba_tab = d * (densite - rho_max) + prob
    return np.where(np.random.binomial(1, proba_tab) == 0)


class Application(tk.Frame):
    """
    Les valeurs propres réelles sont commentées dans les plots car elles ne nous intéressaient pas
    """

    def __init__(self, master=None):
        matplotlib.use('TkAgg')
        tk.Frame.__init__(self, master)

        # ----------------------------------------------------------------------------------------------- Create widgets
        fig = plt.figure(figsize=(6, 6))
        self.ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=False)
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=8)
        self.canvas.draw()

        # Barre de navigation graphique
        toolbar_frame = tk.Frame(master=root)
        toolbar_frame.grid(row=22, column=2)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        # Affichage de l'écriteau + case à remplir pour avoir les données sur le point donnée

        self.sign = tk.Label(root,
                             text="                             Indice dont vous voulez les données:                             ")
        self.sign.grid(row=0, column=0, sticky="n")

        self.entree = tk.Entry(root, justify="right", width=10)
        self.entree.grid(row=0, column=0)  # Positionnement
        self.entree.bind('<Return>', self.valeur)  # Bind de la touche entrée à la simulation

        # Paramètre pour le calcul de la matrice
        self.params = {"taun": ["\u03C4_n", 1],
                       "tauv": ["\u03C4_v", 1],
                       "taur": ["\u03C4_r", 1],
                       "J": ["J", 1],
                       "lambd": ["\u03BB", 0.5]
                       }

        i = 0
        c = 1

        # Initialisation et installation graphique pour les paramètre de la matrice

        for key, ind in self.params.items():
            setattr(self, key, tk.Entry(root, justify="right", width=20))  # Case d'entrée text
            setattr(self, key + "_word", tk.Label(root, text=ind[0] + " :"))  # Ecriteau avec nom de variable
            setattr(self, key + "_bool", tk.BooleanVar())  # Variable correspondant a la case à cochée
            setattr(self, key + "_invariant", tk.Checkbutton(root, variable=getattr(self,
                                                                                    key + "_bool")))  # Ajout du bouton pour la variable au dessus

            # Set la position de chacun des attributs
            getattr(self, key + "_word").grid(row=i, column=c, sticky="n")
            getattr(self, key).grid(row=i, column=c)
            getattr(self, key + "_invariant").grid(row=i, column=c, sticky="w")
            getattr(self, key + "_bool").set(False)  # Set la case sur décochée
            getattr(self, key).insert(0, ind[1])
            getattr(self, key).bind('<Return>', self.matrix)  # Donne les vp de la matrice avec les paramètres écrits
            i += 1

        # Boutton pour explorer les paramètres
        self.exploreparam = tk.Button(root, text="Explore around (non ticked) parameters on the right",
                                      command=self.explore)
        self.exploreparam.grid(row=2, column=0)

        # Boutton pour régler le nombre de point
        self.info_taille = tk.Label(root, text="Points per parameters")
        self.info_taille.grid(row=3, column=0)
        self.taille = tk.Spinbox(root, from_=1, to=100, increment=1)
        self.taille.grid(row=3, column=0, sticky="n")

        # Bouton pour régler la simulation de point aléatoire
        self.randomvalue = tk.Button(root, text="Try for random values", command=self.randomparam)
        self.randomvalue.grid(row=4, column=0, sticky="n")
        self.randomvalue_disclaimer = tk.Label(root,
                                               text='Generates array of size "points per parameters" \n for non-ticked parameters')
        self.randomvalue_disclaimer.grid(row=4, column=0)

        # Bouton pour l'affichage de montecarlo
        self.monte = tk.Button(root, text="MonteCarlo", command=self.montecarlo)
        self.monte.grid(row=5, column=0)

        # Boutons pour regler la quantité de points conservés
        self.decimer_r = tk.Entry(root, justify="right", width=20)
        self.decimer_r.grid(row=6, column=0, sticky="e")
        self.r_label = tk.Label(root, text="Rayon: ")
        self.r_label.grid(row=6, column=0, sticky="w")
        self.decimer_prob = tk.Entry(root, justify="right", width=20)
        self.decimer_prob.grid(row=7, column=0, sticky="e")
        self.prob_label = tk.Label(root, text="Probabilité: ")
        self.prob_label.grid(row=7, column=0, sticky="w")
        self.decimer_button = tk.Button(root, text="Réduire les points", command=self.reduire)
        self.decimer_button.grid(row=8, column=0)

        # Bouton pour lancer la simulation de l'amplitude du cycle final
        self.simulation_button = tk.Button(root, text="Simuler l'amplitude finale", command=self.simuler_amplitude)
        self.simulation_button.grid(row=9, column=0)

        self.dico = []  # Tableau contenant les infos sur les points, les parametres du i eme point sont dans la ieme ligne

        # Initialisation hors boucle des tableaux pour montecarlo
        self.x = []
        self.y = []
        self.parametres = []

        maxi = 0.2  # Valeur maximale de la partie imaginaire: utilité graphique seulement

        # Initialisation de la taille des tableaux et des tableaux des paramètres
        taille = int(self.taille.get())

        taun_tab = np.linspace(0.5, 5, taille)
        tauv_tab = np.linspace(0.5, 5, taille)
        taur_tab = np.linspace(0.5, 5, taille)
        J_tab = np.linspace(0.5, 5, taille)
        lambda_tab = [0.5]

        # Indice du point
        i = 0

        # Boucle avec tout les tableaux (peu optimisé)
        for tau_n in taun_tab:
            for tau_v in tauv_tab:
                for tau_r in taur_tab:
                    for J in J_tab:
                        for lambd in lambda_tab:
                            res = matrice(tau_r, tau_n, tau_v, J, lambd)  # Calcul des vp

                            reelle = res[np.isreal(res)]  # Tableau contenant uniquement les vp réelles

                            if len(reelle) == 3:  # Choix couleur pour les vp : vert si elles sont toutes réelles, bleu sinon
                                color = "green"

                            else:
                                color = "blue"

                            complexe = res[np.iscomplex(res)]  # Tableau contenant uniquement les vp complexes

                            # for vp in reelle:
                            #    vp = np.real(vp)
                            #    plt.scatter(vp, 0, color=color)
                            #    plt.annotate(str(i), (vp, 0), color=color)

                            if len(complexe) == 2:
                                vp = complexe[0]  # Racine complexe conjuguée donc on en garde qu'une
                                plt.scatter(np.real(vp), np.abs(np.imag(vp)), color="red")
                                plt.annotate(str(i), (np.real(vp), np.abs(np.imag(vp))), color="red")

                                if np.abs(np.imag(vp)) > maxi:  # Reset du max pour l'echelle graphique
                                    maxi = np.abs(np.imag(vp))

                            self.dico.append([tau_n, tau_v, tau_r, J, lambd])  # Ajout au dictionnaire des paramètres

                            i = i + 1

        plt.grid()  # Ajout d'une grille
        plt.plot([0, 0], [-0.2, 1.1 * maxi], color="black")  # Droite x = 0
        plt.ylim(-0.2, 1.1 * maxi)
        plt.title("Vert = 3 vp reelles, Bleu = 1 vp reelle, Rouge = vp complexe")

        self.canvas.draw()

    def explore(self, *_):
        """
        A partir des points données en entrée, on calcule des vp dont les paramètres sont compris entre 80% et 120%
        des paramètres initiaux avec un np.linspace
        """

        self.ax.clear()  # Reset le plot

        self.dico = []  # Redefinition d'un dictionnaire
        taille = int(self.taille.get())

        param_tab = [[[], [], [], [], []],
                     [self.taun.get(), self.tauv.get(), self.taur.get(), self.J.get(), self.lambd.get()],
                     ]

        i = 0
        for key, ind in self.params.items():
            boolean = getattr(self, key + "_bool").get()  # Boolean pour savoir la valeur du paramètre est fixée
            value = float(param_tab[1][i])  # valeur du paramètre en question

            if boolean:
                param_tab[0][i] = np.array([value])  # Paramètre fixé

            else:
                param_tab[0][i] = np.linspace(0.8 * value, 1.2 * value,
                                              taille)  # Linspace en 80% et 120% de sa valeur initiale

            i = i + 1

        taun_tab, tauv_tab, taur_tab, J_tab, lambda_tab = param_tab[0]  # Obtention des tableaux de paramètres

        # Copié collé du code dans l'initialisation
        maxi = 0.2
        i = 0

        for tau_n in taun_tab:
            for tau_v in tauv_tab:
                for tau_r in taur_tab:
                    for J in J_tab:
                        for lambd in lambda_tab:
                            res = matrice(tau_r, tau_n, tau_v, J, lambd)

                            reelle = res[np.isreal(res)]

                            if len(reelle) == 3:
                                color = "green"

                            else:
                                color = "blue"

                            complexe = res[np.iscomplex(res)]

                            # for vp in reelle:
                            #    vp = np.real(vp)
                            #    plt.scatter(vp, 0, color=color)
                            #    plt.annotate(str(i), (vp, 0), color=color)

                            if len(complexe) == 2:
                                vp = complexe[0]
                                plt.scatter(np.real(vp), np.abs(np.imag(vp)), color="red")
                                plt.annotate(str(i), (np.real(vp), np.abs(np.imag(vp))), color="red")

                                if np.abs(np.imag(vp)) > maxi:
                                    maxi = np.abs(np.imag(vp))

                            self.dico.append([tau_n, tau_v, tau_r, J, lambd])

                            i = i + 1

        plt.grid()
        plt.plot([0, 0], [-0.2, 1.1 * maxi], color="black")
        plt.ylim(-0.2, 1.1 * maxi)
        plt.title("Vert = 3 vp reelles, Bleu = 1 vp reelle, Rouge = vp complexe")

        self.canvas.draw()

    def randomparam(self, *_):
        """"
        Permet d'obtenir des vp avec des paramètres aléatoires (mais dans une fenêtre définie)
        """
        self.ax.clear()
        self.dico = []
        taille = int(self.taille.get())

        param_tab = [[[], [], [], [], []],
                     [self.taun.get(), self.tauv.get(), self.taur.get(), self.J.get(), self.lambd.get()],
                     [2, 5, 10, 10, 1]  # Les paramètres sont dans l'intervalle [0, Ce nombre[
                     ]

        i = 0
        for key, ind in self.params.items():  # Comme précédemment dans self.explore
            boolean = getattr(self, key + "_bool").get()

            if boolean:
                param_tab[0][i] = np.array([float(param_tab[1][i])])

            else:
                param_tab[0][i] = np.round(param_tab[2][i] * np.random.rand(taille),
                                           2)  # Tableau de nombre aléatoire distribuée uniformément

            i = i + 1

        taun_tab, tauv_tab, taur_tab, J_tab, lambda_tab = param_tab[0]

        # Copié coller de l'initialisation
        maxi = 0.2
        i = 0

        for tau_n in taun_tab:
            for tau_v in tauv_tab:
                for tau_r in taur_tab:
                    for J in J_tab:
                        for lambd in lambda_tab:
                            res = matrice(tau_r, tau_n, tau_v, J, lambd)

                            reelle = res[np.isreal(res)]

                            if len(reelle) == 3:
                                color = "green"

                            else:
                                color = "blue"

                            complexe = res[np.iscomplex(res)]

                            # for vp in reelle:
                            #    vp = np.real(vp)
                            #    plt.scatter(vp, 0, color=color)
                            #    plt.annotate(str(i), (vp, 0), color=color)

                            if len(complexe) == 2:
                                vp = complexe[0]
                                plt.scatter(np.real(vp), np.abs(np.imag(vp)), color="red")
                                plt.annotate(str(i), (np.real(vp), np.abs(np.imag(vp))), color="red")

                                if np.abs(np.imag(vp)) > maxi:
                                    maxi = np.abs(np.imag(vp))

                            self.dico.append([tau_n, tau_v, tau_r, J, lambd])

                            i = i + 1

        plt.grid()
        plt.plot([0, 0], [-0.2, 1.1 * maxi], color="black")
        plt.ylim(-0.2, 1.1 * maxi)
        plt.title("Vert = 3 vp reelles, Bleu = 1 vp reelle, Rouge = vp complexe")

        self.canvas.draw()

    def valeur(self, *_):
        """"
        Permet d'obtenir les données d'un point i
        """
        i = int(self.entree.get())
        if i >= len(self.dico):  # Si l'entrée est plus grande que la taille du dico => erreur
            string = "Invalid value"
            a, b, c = 0, 0, 0

        else:
            tau_n, tau_v, tau_r, J, lambd = np.round(self.dico[i],
                                                     2)  # Obtention des paramètres (arrondis à la deuxieme décimale)
            string = "\u03C4_n = " + str(tau_n) + ", \u03C4_v = " + str(tau_v) + ", \u03C4_r = " + str(
                tau_r) + ", J = " + str(J) + ", \u03BB = " + str(lambd)
            # string contenant les noms et valeurs des paramètres

            a, b, c = matrice(tau_r, tau_n, tau_v, J, lambd)  # a, b et c sont les vp de la matrice

        # Reset de l'affichage des paramètres et vp
        params = tk.Label(root,
                          text="                                                                                                         ")
        params.grid(row=1, column=0, sticky="n")
        vp = tk.Label(root,
                      text="                                                                                                         ")
        vp.grid(row=1, column=0, sticky="s")

        # Affichage paramètres et vp
        params = tk.Label(root, text=string)
        params.grid(row=1, column=0, sticky="n")
        vp_str = "Les valeurs propres sont " + str(np.round(a, 2)) + ", " + str(np.round(b, 2)) + ", " + str(
            np.round(c, 2))
        vp = tk.Label(root, text=vp_str)
        vp.grid(row=1, column=0, sticky="s")

    def matrix(self, *_):
        """"
        Affiche les valeurs propres de la matrice en fonction des paramètres d'entrée
        """
        tau_n, tau_v, tau_r, J, lambd = float(self.taun.get()), float(self.tauv.get()), float(self.taur.get()), float(
            self.J.get()), float(self.lambd.get())

        a, b, c = matrice(tau_r, tau_n, tau_v, J, lambd)

        vp_str = "Les valeurs propres sont " + str(np.round(a, 2)) + ", " + str(np.round(b, 2)) + ", " + str(
            np.round(c, 2))

        # Reset de l'affichage des vp
        vp = tk.Label(root,
                      text="                                                                                           ")
        vp.grid(row=5, column=1)

        vp = tk.Label(root, text=vp_str)
        vp.grid(row=5, column=1)

    def montecarlo(self, *_):
        """"
        Permet d'afficher un graphe contenant les vp complexe de partie réelle positive et ce toute sur un même graphe
        A noter que rappuyer plusieurs fois sur le bouton rajoute des points sans supprimer les autres.
        """
        self.ax.clear()
        lenght = 1000  # Nombre de points simulés

        # Tableau de 5 colonne contenant taur, taun, tauv, J et lambda et de lenght colonne
        random = np.array([10 * np.random.rand(lenght),
                           1 * np.random.rand(lenght),
                           10 * np.random.rand(lenght) + 0.0000001,
                           20 * np.random.rand(lenght) + 0.0000001,
                           np.random.rand(lenght)])

        maxi = 1 / 1.1

        for i in range(lenght):
            vp = matrice(random[0][i], random[1][i], random[2][i], random[3][i], random[4][i])  # Calcul des vp
            vp = vp[np.iscomplex(vp)]  # Verification qu'il y a deux vp complexe

            if len(vp) != 0:
                if np.real(vp[0]) >= 0:
                    self.x.append(np.real(vp[0]))
                    self.y.append(
                        np.abs(np.imag(vp[0])))  # Ajout de la vp complexe avec la partie réelle positive au graphe
                    self.parametres.append(random[:, i])  # Sauvegarde des paramètres utilisées pour obtenir ces vp
                    if np.abs(np.imag(vp[0])) > maxi:  # Reset pour la fenetre graphique
                        maxi = np.abs(np.real(vp[1]))

        plt.scatter(self.x, self.y, color="red")
        plt.plot([0, 0], [-0.2, 1.1 * maxi], color="black")
        plt.ylim(0, 1.1 * maxi)
        plt.title("MonteCarlo")

        self.canvas.draw()

    def reduire(self, *_):
        """
        Permet de réduire les points sur la figure de montecarlo à partir de la fonction décimer
        Utile pour faire du calcul sur un nombre réduit de point qui définisse la conique avec des vp complexes
        """

        mes_pos = decimer([self.x, self.y], float(self.decimer_r.get()), float(self.decimer_prob.get()))
        self.parametres = np.array(self.parametres)[mes_pos]
        self.x = np.array(self.x)[mes_pos]
        self.y = np.array(self.y)[mes_pos]

        plt.clf()
        maxi = 1 / 1.1
        plt.scatter(self.x, self.y, color="red")
        plt.plot([0, 0], [-0.2, 1.1 * maxi], color="black")
        plt.ylim(0, 1.1 * maxi)
        plt.title("MonteCarlo")

        self.canvas.draw()

    def simuler_amplitude(self, *_):
        """
        Permet d'obtenir l'amplitude d'un cycle final à partir d'un jeu de paramètres initiaux
        """

        np.save("Params_simu", self.parametres)
        np.save("x_simu", self.x)
        np.save("y_simu", self.y)

        x0 = 4.7
        y0 = 0.0
        vx0 = 0.0
        vy0 = 0.0
        theta0 = 0.1
        omega0 = 0.0

        kappa = 10.0
        sigma = 5.0
        beta = 0.0
        N_w = 0.0
        bruit = 0.0

        t_max = 500
        nb_points = 5000

        amp = np.zeros(len(self.x))
        print(len(self.x))
        for i in range(len(self.x)):
            if i % 10 == 0:
                print(i)
            tau_r, tau_n, tau_v, J, lambd = self.parametres[i]
            t, sol = ode_solution([x0, y0, vx0, vy0, theta0, omega0,
                                   tau_v, kappa, sigma, tau_n, beta, J, lambd, tau_r, N_w, t_max, nb_points,
                                   bruit])

            theta = sol[:, 4]

            amp[i] = np.max(theta[1000:])

        np.save("amplitude_simu", np.array(amp))

        plt.clf()
        maxi = 1 / 1.1
        plt.scatter(x=self.x, y=self.y, c=np.array(amp), cmap='plasma', vmin=0, vmax=3)
        for i in range(len(self.x)):
            plt.annotate(str(i), (self.x[i], self.y[i]), color="black")

        plt.colorbar(label="Amplitude du cycle limite (rad)", orientation="vertical")
        plt.plot([0, 0], [-0.2, 1.1 * maxi], color="black")
        plt.ylim(0, 1.1 * maxi)
        plt.title("Amplitude du cycle final lorsqu'il existe")

        self.canvas.draw()


#Démarrage du code
if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
