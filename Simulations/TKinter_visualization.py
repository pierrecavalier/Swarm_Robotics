import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from RK_ODE import ode_solution
from get_trajectory import traj
import os
import matplotlib.cm as cm
import matplotlib.colors


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.)  # XXX assume int() truncates!
    f = (h * 6.) - i
    p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


def transfo_fourier(theta, x):  # Calcul la transformée de fourier d'angle
    """
    Effectue une transformée de Fourier de theta(x) et renvoit la norme de la transformée en fonction de la fréquence
    shiftée
    """
    theta = (theta + np.pi) % (2 * np.pi)
    theta -= theta.mean()
    f_x = np.fft.fftfreq(x.size, d=x[1] - x[0])
    nu = np.fft.fftshift(f_x)
    tf_x = np.fft.fft(theta)
    psd = np.abs(tf_x)
    psd = np.fft.fftshift(psd)
    return nu, psd


def combine_funcs(*funcs):  # Permet de combiner les fonction données en entrée
    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)

    return combined_func


class Application(tk.Frame):
    """"Visualizer for the solutions of the ODE system written in RK_ODE.py"""

    def __init__(self, master=None, rainbow=True):
        matplotlib.use('TkAgg')
        tk.Frame.__init__(self, master)

        # ----------------------------------------------------------------------------------------------- Create widgets
        fig = plt.figure(figsize=(6, 6))
        self.ax2 = fig.add_axes([0.85, 0.1, 0.08, 0.8], polar=False)
        self.ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=False)
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=4, rowspan=10)
        self.canvas.draw()

        # Bare de navigation graphique
        toolbar_frame = tk.Frame(master=root)
        toolbar_frame.grid(row=22, column=4)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        # Conservation du Zoom pour les figure : Bordures de la fenetre initiale

        self.modepre = -1
        self.left = -14
        self.right = 14
        self.up = 14
        self.down = -14

        self.chunks_p = np.array([])

        # Format : nom de la variable : min, max, pas, nom affiché, valeur initiale
        # Le min max et pas n'ont pas d'importance, ce sont des artéfacts de l'ancien code

        # args = parametres de la fonction odeint

        self.args = {"x_0": [-2, 2, 0.1, "X_0", 0],
                     "y_0": [-2, 2, 0.1, "Y_0", 0],
                     "vx_0": [-2, 2, 0.1, "V0_X", 0],
                     "vy_0": [-2, 2, 0.1, "V0_Y", 0],
                     "theta_0": [-3.2, 3.2, 0.01, u"\u03B8_0", 0],
                     "omega_0": [-2, 100, 0.1, u"\u03C9_0", 0],
                     "tau_vsl": [-2, 2, 0.01, u"\u03C4_V", 1],
                     "kappa": [0, 10, 0.1, u"\u03BA", 10],
                     "sigma": [0, 20, 0.01, u"\u03C3", 5],
                     "tau_nsl": [-2, 2, 0.1, u"\u03C4_N", 1],
                     "beta": [-0.1, 0.1, 0.001, u"\u03B2", 0],
                     "I": [0, 10, 0.01, "I", 1],
                     "lambd": [0, 10, 0.1, u"\u03BB", 0],
                     "tau_r": [0, 10, 0.1, u"\u03C4_R", 0],
                     "N_w": [0, 10, 0.1, u"N_w", 0],
                     "t_max": [20, 200, 1, "T_MAX", 20],
                     "nb_points": [50, 1000, 10, "NB_POINTS", 50],
                     "noise": [0, 1, 0.01, "NOISE", 0],
                     }

        # paramfourier = parametres de la fonction pour régler la transformée de Fourier

        self.paramfourier = {"debutfourier": [0, 100, 1, "DEBUT FOURIER", 0],
                             "finfourier": [0, 100, 1, "FIN FOURIER", 100]
                             }

        # params = parametres purement graphiques

        self.params = {"start_meas": [0, 100, 0.1, "START_OF_MEASURE", 0],
                       "end_meas": [0, 100, 0.1, "END_OF_MEASURE", 0],
                       "scale": [10, 500, 10, "SCALE", 200],
                       "offset_x": [-20, 20, 0.1, "OFFSET X", 0],
                       "offset_y": [-20, 20, 0.1, "OFFSET Y", 0],
                       "arr_length": [0, 5, 0.1, "ARR_LENGHT", 0.5],
                       "speed": [0, 5, 0.1, "VEL_LENGHT", 0],
                       "pourcentVisible": [1, 15, 1, "STEP GRAPHIQUE", 1],
                       }

        # Ajoute un attribut dans chaque dictionnaire qui permet d'avoir une case blanche à remplir

        largeur = 20

        for key, ind in self.args.items():
            setattr(self, key, tk.Entry(root, justify="right", width=largeur))
            setattr(self, key + "_word", tk.Label(root, text=ind[3] + " :"))

        for key, ind in self.paramfourier.items():
            setattr(self, key, tk.Entry(root, justify="right", width=largeur))
            setattr(self, key + "_word", tk.Label(root, text=ind[3] + " :"))

        for key, ind in self.params.items():
            setattr(self, key, tk.Entry(root, justify="right", width=largeur))
            setattr(self, key + "_word", tk.Label(root, text=ind[3] + " :"))

        # Ajout de fenetres pour les coches

        self.fen = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.fen1 = tk.PanedWindow(root, orient=tk.VERTICAL)
        self.fen2 = tk.PanedWindow(root, orient=tk.VERTICAL)
        self.fen.add(self.fen1)
        self.fen.add(self.fen2)

        # Creation des variables pour afficher les simulations/angles graphique
        self.bool_theta = tk.BooleanVar()
        self.bool_V = tk.BooleanVar()
        self.bool_Sim = tk.BooleanVar()
        self.bool_Mes = tk.BooleanVar()

        # Definition des coches
        self.fen1.add(tk.Checkbutton(self.fen1, text=u"Plot \u03B8", variable=self.bool_theta, command=self.plot))
        self.bool_theta.set(True)

        self.fen1.add(tk.Checkbutton(self.fen1, text="Plot \u03B1", variable=self.bool_V, command=self.plot))
        self.bool_V.set(False)

        self.fen2.add(tk.Checkbutton(self.fen2, text="Plot Simulation", variable=self.bool_Sim, command=self.plot))
        self.bool_Sim.set(True)

        self.fen2.add(tk.Checkbutton(self.fen2, text="Plot Mesure", variable=self.bool_Mes, command=self.plot))
        self.bool_Mes.set(False)

        self.fen.grid(row=6, column=2)

        # Ajout bouton de chargement/téléchargement puis réglage de leur position

        self.download = tk.Button(root, text="Save parameters", command=self.save)
        self.loadparam = tk.Button(root, text="Load parameters", command=self.load)
        self.loadchunk = tk.Button(root, text="Data to load", command=self.import_chunk)

        self.download.grid(row=8, column=2)
        self.loadparam.grid(row=9, column=2)
        self.loadchunk.grid(row=7, column=2)

        # Ajout d'une spinbox pour pouvoir tourner les mesures réelles

        mur = tk.IntVar()
        self.mur_word = tk.Label(root, text="Rotate measure")
        self.mur = tk.Spinbox(root, textvariable=mur, values=[0, 90, 180, -90], command=self.plot, wrap=True)
        self.mur_word.grid(row=4, column=2, sticky="n")
        self.mur.grid(row=5, column=2, sticky="s")

        # Positionnement des zone de saisie pour les paramètres dans les listes dans l'ordre :
        # argument - paramfourier - parametres

        maxrow = 11
        i = 0
        c = 0
        for key, ind in self.args.items():
            getattr(self, key + "_word").grid(row=i, column=c, sticky="n")  # Mot clé correspondant
            getattr(self, key).grid(row=i, column=c)  # Positionnement
            getattr(self, key).insert(0, ind[4])  # Valeur initiale
            getattr(self, key).bind('<Return>', self.simulate)  # Bind de la touche entrée à la simulation
            i += 1
            if i == maxrow:
                i = 0
                c += 1

        for key, ind in self.paramfourier.items():
            getattr(self, key + "_word").grid(row=i, column=c, sticky="n")
            getattr(self, key).grid(row=i, column=c)
            getattr(self, key).insert(0, ind[4])
            getattr(self, key).bind('<Return>', combine_funcs(self.fourier, self.plot))
            i += 1
            if i == maxrow:
                i = 0
                c += 1

        for key, ind in self.params.items():
            getattr(self, key + "_word").grid(row=i, column=c, sticky="n")
            getattr(self, key).grid(row=i, column=c)
            getattr(self, key).insert(0, ind[4])
            getattr(self, key).bind('<Return>', self.plot)
            i += 1
            if i == maxrow:
                i = 0
                c += 1

        # Initialition des variables self. en dehors des fonctions (Peu utile en pratique)

        self.sol = np.zeros((5, 5))  # Solution de l'EDO de la forme [rx, ry, vx, vy, theta, omega]
        self.t = np.array([0, 1])  # Tableau de temps de la solution initialisé avec deux elements pour fourier

        self.freq = 0  # Frequence maximale dans la transformée de Fourier
        self.fouriery = []  # Tableau des normes de la transformée de Fourier
        self.fourierx = []  # Tableau des pulsations de la transformée de Fourier
        self.data = []  # Tableau contenant theta
        self.tfourier = []  # Tableau de temps tronqué par debutfourier et finfourier
        self.ang_vit = []  # Tableau des angles fait par v et la droite des abscisses
        self.path = "../Wall_data"  # Chemin d'accès au fichier .txt

        # Curseur pour choisir ce qu'il faut afficher dans la fenetre de plot

        self.mode = tk.Scale(root, from_=0, to=4, resolution=1, length=200, orient=tk.HORIZONTAL,
                             label=u"TRAJ. - VEL. - ANGLE - FT - PHASE",
                             command=self.plot)

        self.mode.set(0)
        self.mode.grid(row=10, column=2)

        self.rainbow = rainbow

    def simulate(self, *_):
        """
        Calcul la solution fourier et l'affiche
        """
        self.calcul_sol()
        self.fourier()
        self.plot()

    def save(self):
        """
        Indiquer un fichier dans lequel sera sauvegarder un .txt contenant les parametres
        Choix du nom du fichier, sauvegarde dans Saved_Files sous le nom donné.txt avec un (1)
        si déjà pris et ainsi de suite ( (2), (3), ...)
        """
        answer = tk.simpledialog.askstring("Input", "Nom du fichier", parent=root)

        if os.path.exists("../Saved_files/" + answer + ".txt"):
            rename = tk.messagebox.askquestion("WARNING", "Un fichier portant le même nom existe déjà, " +
                                               "souhaitez-vous tout de même l'écraser ? " +
                                               "Attention, cette action est irréversible.",
                                               icon="warning"
                                               )
            if rename == "no":
                i = 0
                temp = answer
                while os.path.exists("../Saved_files/" + temp + ".txt"):
                    i += 1
                    temp = answer
                    temp = temp + str(i)

                answer = answer + "(" + str(i) + ")"

        # Ouverture du fichier

        f = open("../Saved_files/" + answer + ".txt", "w+")

        # Ecriture ligne par ligne des args puis fourierparams puis params dans l'ordre du tableau

        for key, ind in self.args.items():
            f.write(key + " = " + str(getattr(self, key).get()) + "\n")

        for key, ind in self.params.items():
            f.write(key + " = " + str(getattr(self, key).get()) + "\n")

        for key, ind in self.paramfourier.items():
            f.write(key + " = " + str(getattr(self, key).get()) + "\n")

        # Ecriture angle de rotation des données réelles
        f.write("Angle de rotation = " + str(self.mur.get()) + "\n")

        # Si des données sont chargées leur adresse est sauvegardé
        if self.chunks_p.any:
            f.write("Adresse du fichier chargé: " + str(self.path))
        f.close()

    def load(self):
        """
        load un .txt obtenu à partir de self.save et charge ces parametres. Attention aux versionning (Un changement
        de paramètres et tout plante)
        """
        file = tk.filedialog.askopenfilename(initialdir="../Saved_files/", title="Select file",
                                             filetypes=(("text files", "*.txt"), ("all files", "*.*")))

        # Ouverture du fichier
        data = open(file, "r")

        # Reordonnement du tableau obtenu pour ne garder que le float
        tab = data.readlines()

        argument = np.zeros_like(tab)

        for i in range(len(tab)):
            argument[i] = ((tab[i].split()[-1]).split("\n")[0])

        # Changement de valeur dans chaque case

        i = 0
        for key in self.args.keys():
            getattr(self, key).delete(0, 'end')
            getattr(self, key).insert(0, argument[i])
            i += 1

        for key in self.params.keys():
            getattr(self, key).delete(0, 'end')
            getattr(self, key).insert(0, argument[i])
            i += 1

        for key in self.paramfourier.keys():
            getattr(self, key).delete(0, 'end')
            getattr(self, key).insert(0, argument[i])
            i += 1

        self.mur.delete(0, 'end')
        self.mur.insert(0, argument[i])

        chemin = argument[-1]

        if len(chemin) > 30:  # Condition pour savoir si la dernière ligne est une adresse. (sinon chemin = self.mur
            # qui est un angle en degré donc moins de 30 caractères.

            self.path = ".." + chemin.split("swarm-robotics-active-matter")[-1]

            self.chunks_p = traj(self.path)

    def import_chunk(self):  # Demande à l'utilisateur un fichier et le charge
        """
        Import un chunck (.csv) pour le lire
        """
        self.path = tk.filedialog.askopenfilename(initialdir="../wall_csv_0707/", title="Select file",
                                                  filetypes=(("CSV", "*.csv"), ("all files", "*.*")))

        self.chunks_p = traj(self.path)

    def calcul_sol(self, *_):
        """
        Calcul la solution de l'EDO à partir du fichier RK_ODE.py
        """
        arg = list()
        for key in self.args.keys():
            value = getattr(self, key).get()
            if value == '':
                value = 0

            arg.append(float(value))

        t, sol = ode_solution(arg)

        self.t = t  # Tableau de temps de la solution

        self.sol = sol  # Tableau de la solution de la forme [rx, ry, vx, vy, theta, omega]

    def fourier(self, *_):
        """
        Calcul la transformée de Fourier pour les données simulées
        """

        # Obtention des paramètres nécessaire pour une transformée de Fourier

        n = int(self.nb_points.get())
        sol = self.sol
        t = self.t

        theta = sol[:, 4]

        vx, vy = sol[:, 2], sol[:, 3]

        debutfourier = int(int(self.debutfourier.get()) * n / 100)
        finfourier = int(int(self.finfourier.get()) * n / 100)

        # Calcul de la transformée de Fourier en prenant compte des bornes

        tempsfourier = []
        psd = []
        nu = []
        ang_vit = []

        if len(theta[debutfourier:finfourier]) >= 2:
            theta = theta[debutfourier:finfourier]
            vx = vx[debutfourier:finfourier]
            vy = vy[debutfourier:finfourier]
            tempsfourier = t[debutfourier:finfourier]

            nu, psd = transfo_fourier(theta, tempsfourier)
            ang_vit = np.angle((vx + 1j * vy))

        self.data = theta
        self.ang_vit = ang_vit
        self.tfourier = tempsfourier
        self.freq = 0

        # Recherche de la fréquence maximale puis normalisation
        string = ""
        if np.max(np.abs(psd)) != 0:
            self.freq = np.abs(nu[np.abs(psd).argmax()])
            psd = psd / np.max(psd)
            self.fouriery = psd
            self.fourierx = nu

            string = "La fréquence angulaire vaut " + str(np.round(self.freq, 2)) + " v0/d0."

        # Affichage de la fréquence angulaire
        frequence = tk.Label(root, text=string)
        # frequence.grid(row=6, column=2)

    def plot(self, *_):
        """
        Affiche la solution de l'EDO au préalable calculée par calcul_sol
        Permet d'afficher différents graphes sans tout recalculer
        """
        # Initialisation pour la colorbar
        treal = 0
        my_end = 0
        my_start = 0

        # Conservation du zoom si affichage de la trajectoire

        if self.modepre == 4 and self.mode.get() != 4:
            for ax in plt.gcf().axes:
                plt.delaxes(ax)
            fig = plt.gcf()
            self.ax2 = fig.add_axes([0.85, 0.1, 0.08, 0.8], polar=False)
            self.ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=False)

        if self.modepre != 0:
            left, right = self.left, self.right
            down, up = self.down, self.up
        else:
            left, right = plt.xlim()
            down, up = plt.ylim()

            self.left, self.right = left, right
            self.down, self.up = down, up

        if int(self.mode.get()) != 4:
            self.ax.clear()  # clear axes from previous plot
            self.ax2.cla()
            self.ax2.set_frame_on(False)
            self.ax2.set_axis_off()

        x, y, theta = np.array([]), np.array([]), np.array([])

        # Affichage données réelles
        if self.chunks_p.any():
            my_start = int(self.chunks_p[3, int(self.start_meas.get())])
            my_end = int(self.chunks_p[3, int(self.end_meas.get())])

            x = self.chunks_p[0, my_start:my_end]
            y = self.chunks_p[1, my_start:my_end]
            theta = self.chunks_p[2, my_start:my_end]
            nframe = self.chunks_p[3, my_start:my_end]

            framerate = 15  # FPS

            treal = nframe / framerate

            # Rotation selon le mur sur lequel la mesure à été effectuée

            d0 = -np.pi / 2

            theta = theta + d0

        # Rotation des mesures à partir de la spinbox

        angle_rot = np.deg2rad(float(self.mur.get()))

        matrix_rot = [[np.cos(angle_rot), -np.sin(angle_rot)], [np.sin(angle_rot), np.cos(angle_rot)]]

        theta = ((theta + angle_rot) + np.pi) % (2 * np.pi) - np.pi
        x, y = np.dot(matrix_rot, [x, y])

        # Recuperation solution et quel graphe doit être affiché

        sol = self.sol

        md = int(self.mode.get())
        speed = float(self.speed.get())

        # -------------------------------------------------------------------------------------- Affichage de la position

        if md == 0:

            # plot l'arene
            sigma = float(self.sigma.get())

            if sigma != 0:
                toplot = np.array([np.minimum(3 * down, -10), np.maximum(3 * up, 10)])
                toplotbis = np.array([sigma, sigma])

                # toplot = np.array([sigma, sigma])
                # toplotbis = np.array([sigma, -sigma])

                plt.plot(toplotbis, toplot, color="black")
                # plt.plot(toplotbis, -toplot, color="black")
                # plt.plot(toplot, toplotbis, color="black")
                # plt.plot(-toplot, toplotbis, color="black")

            length = float(self.arr_length.get())
            mod = int(self.pourcentVisible.get())

            # Affichage données réelles

            if len(theta) != 0:

                off_x = float(self.offset_x.get())
                off_y = float(self.offset_y.get())
                scale = float(self.scale.get())

                if self.bool_Mes.get():

                    norm = matplotlib.colors.Normalize(treal[0], treal[-1])
                    self.ax2.set_frame_on(True)
                    self.ax2.set_axis_on()
                    self.cb = plt.colorbar(cm.ScalarMappable(norm, "hsv"), cax=self.ax2, label="Time (s)", shrink=0.8)

                    for i in range(0, len(x), mod):
                        col = hsv_to_rgb(i / len(x), 1, 0.7) if self.rainbow else "b"

                        # Position
                        plt.plot(1.0 / scale * x[i] + off_x,
                                 1.0 / scale * y[i] + off_y,
                                 "+", color=col)

                        # Vecteur n
                        plt.arrow(1.0 / scale * x[i] + off_x,
                                  1.0 / scale * y[i] + off_y,
                                  np.cos(theta[i]) * length, np.sin(theta[i]) * length, color=col)

            # Affichage données simulées
            if self.bool_Sim.get():
                for i in range(0, len(sol), mod):
                    col = hsv_to_rgb(i / len(sol), 1, 1) if self.rainbow else "b"
                    if sol[i, 0] < 10 ** 2 and sol[i, 1] < 10 ** 2:
                        # Position
                        plt.plot(sol[i, 0], sol[i, 1], "+", color=col)

                        # Vecteur n
                        plt.arrow(sol[i, 0], sol[i, 1], np.array(sol[i, 2]) * speed,
                                  np.array(sol[i, 3]) * speed, color="black")

                        # Vecteur vitesse
                        plt.arrow(sol[i, 0], sol[i, 1], np.array(np.cos(sol[i, 4])) * length,
                                  np.array(np.sin(sol[i, 4])) * length, color=col)

            plt.title("Position de la particule en fonction du temps")
            plt.xlabel("x [d0]")
            plt.ylabel("y [d0]")

            plt.xlim(left, right)
            plt.ylim(down, up)

        # --------------------------------------------------------------------------------------- Affichage de la vitesse

        elif md == 1:
            plt.grid()
            ti = np.linspace(0, float(self.t_max.get()), int(self.nb_points.get()))

            if self.bool_Sim.get():
                plt.plot(ti, np.sqrt(sol[:, 2] ** 2 + sol[:, 3] ** 2), label="Simulation")
                plt.legend()

            tmes = np.linspace(0, float(self.t_max.get()), len(x))

            if self.bool_Mes.get():
                if float(my_end - my_start) != 0:
                    # Calcul de la norme de la vitesse en utilisant le gradient
                    plt.plot(tmes, np.sqrt(np.gradient(x) ** 2 + np.gradient(y) ** 2), label="Chunk")
                    plt.legend()

            plt.title("Vitesse en fonction du temps")
            plt.xlabel("Temps [d0/v0]")
            plt.ylabel("Vitesse [v0]")

        # ----------------------------------------------------------------------------------------- Affichage de l'angle

        elif md == 2:
            plt.grid()
            if float(self.finfourier.get()) - float(self.debutfourier.get()) > 0:
                if self.bool_Sim.get():
                    if self.bool_theta.get():  # Theta simulé
                        plt.plot(self.tfourier, self.data, label=u"\u03B8 simulé")
                        plt.legend()

                    if self.bool_V.get():  # Alpha simulé
                        plt.plot(self.tfourier, self.ang_vit, label=u"\u03B1 simulé")
                        plt.legend()

            if self.bool_Mes.get():
                if float(my_end - my_start) != 0:

                    tmes = np.linspace(0, float(self.t_max.get()), len(x))

                    if self.bool_theta.get():  # Theta mesuré
                        plt.plot(tmes, theta, label=u"\u03B8 mesuré")
                        plt.legend()

                    if self.bool_V.get() and len(x) > 1:  # Alpha mesuré
                        vx = np.gradient(x)
                        vy = np.gradient(y)
                        vit_angle = np.angle(vx + 1j * vy)

                        plt.plot(tmes, vit_angle, label=u"\u03B1 vitesse mesuré")
                        plt.legend()

            plt.title("Angle de n et de la vitesse en fonction du temps")
            plt.xlabel("Temps [d0/v0]")
            plt.ylabel("Angle [rad]")

        # ----------------------------------------------------------------------- Affichage de la transformée de Fourier

        elif md == 3:
            plt.grid()
            abscisse = list(self.fourierx)
            ordonne = list(np.abs(self.fouriery))

            if len(abscisse) != 0:
                abscisse.append(np.abs(abscisse[0]))
                ordonne.append(ordonne[0])

            plt.plot(abscisse, ordonne, label="Simulation")
            plt.title("Module de la transformée de Fourier en fonction de la fréquence")
            plt.xlabel("Fréquence [d0/v0]")
            plt.ylabel("Module de la transformée de Fourier [v0/d0]")

            """ Transformée de Fourier pour les données mesurées (infaisable car pas les mêmes unités)
            tmes = np.linspace(0, self.t_max.get(), len(x))
            debutfourier = int(self.debutfourier.get() * len(x))
            finfourier = int(self.finfourier.get() * len(x))
            psd = []
            nu = []

            if finfourier - debutfourier > 1:
                theta = theta[debutfourier:finfourier]
                tmes = tmes[debutfourier:finfourier]

                nu, psd = transfo_fourier(theta, tmes)
            plt.plot(nu, psd, label = "chunk")
            """
            plt.legend()

        # ----------------------------------------------------------------------------- Affichage de l'espace des phases

        elif md == 4:

            col = [hsv_to_rgb(i / len(sol[:, 0]), 1, 0.7) for i in range(len(sol[:, 0]))]

            vtheta = plt.subplot(2, 2, 1)
            omegatheta = plt.subplot(2, 2, 2)
            omegav = plt.subplot(2, 2, 3)
            plot3d = plt.subplot(2, 2, 4, projection="3d")

            vtheta.plot(sol[:, 3], sol[:, 4])
            vtheta.set_title(r"$\theta $ en fonction de $v_y$")
            vtheta.set_xlabel(r"$v_y$")
            vtheta.set_ylabel(r"$\theta$")

            omegatheta.plot(sol[:, 5], sol[:, 3])
            omegatheta.set_title(r"$ \theta $ en fonction de $ \omega $")
            omegatheta.set_xlabel(r"$\theta$")
            omegatheta.set_ylabel(r"$\omega$")

            omegav.plot(sol[:, 5], sol[:, 1])
            omegav.set_title(r"$v_y$ en fonction de $ \omega $")
            omegav.set_xlabel(r"$v_y$")
            omegav.set_ylabel(r"$\omega$")

            plot3d.plot3D(sol[:, 3], sol[:, 4], sol[:, 5])
            plot3d.set_xlabel(r"$v_y$")
            plot3d.set_ylabel(r"$\theta$")
            plot3d.set_zlabel(r"$\omega$")

            plt.tight_layout()

        # Permet de garde le zoom
        if self.mode.get() != -1:
            self.modepre = self.mode.get()
        else:
            self.modepre = -1

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root, rainbow=True)
    app.mainloop()
