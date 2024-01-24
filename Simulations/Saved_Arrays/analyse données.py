import matplotlib.pyplot as plt
import numpy as np

x  = np.load("x_simu.npy")
y  = np.load("y_simu.npy")
params = np.load("Params_simu.npy")
amp = np.load("amplitude_simu.npy")

nartefact=np.where(amp >= -0.01)

print(params[345])
fig = plt.figure()
maxi = 1 / 1.1
plt.scatter(x=x[nartefact], y=y[nartefact], c=np.array(amp)[nartefact], cmap='plasma', vmin=0, vmax=3)
#for i in range(len(x)):
#    plt.annotate(str(i), (x[i], y[i]), color="black")

plt.colorbar(label="Amplitude du cycle limite (rad)", orientation="vertical")
plt.plot([0, 0], [-0.2, 1.1 * maxi], color="black")
plt.ylim(0, 1.1 * maxi)
plt.title("Amplitude du cycle limites pour les valeurs propres pour lesquelles il existe")
plt.xlabel("Partie réelle")
plt.ylabel("Partie Imaginaire")
plt.show()