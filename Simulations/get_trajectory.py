import pickle
import numpy as np
from Analysis.Trajectories import load_smooth_trajectory
import matplotlib.pyplot as plt


def load_traj(path):
    f = open(path,'rb')
    my_chunks = pickle.load(f)
    res = list()
    for chunk in my_chunks:
        x,y,theta = np.array(chunk[0]),np.array(chunk[1]),np.array(chunk[2])
        res.append(np.array([x,y,theta]))
    return res

data = load_traj("../Wall_data/KB5_wall3_xyTh.pickle")

def traj(path):
    a = np.genfromtxt(path,delimiter=",", skip_header=1,missing_values="nan",filling_values=np.NaN)
    return np.array([a[:,2],a[:,3],a[:,4],a[:,0]]) #Tableau x,y,theta,numéro de frame

