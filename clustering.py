# -*- coding: utf-8 -*-
"""clustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AKSorDfhZqcQKuYThSvRCshgqI9AcMt3

#kmeans
"""

# Commented out IPython magic to ensure Python compatibility.
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd

file = "..."
x = np.load(file)

def plot_inertia(x) :
  
  inerties = []

  for k in range(2, 15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    inerties.append(kmeans.inertia_)

  #graphique des inerties 
  fig = plt.figure(figsize=(10,5))
  plt.plot(range(2,15), inerties)
  plt.xlabel("Nombres de clusters")
  plt.ylabel("Inertie")
  plt.title("Inertie en fonction du nombre de classes")

def plot_classes(x):
  #création du modèle à 5 classes
  modele_km = KMeans(n_clusters=5)
  modele_km.fit(x)
  KMeans(n_clusters=5, init='k-means++', n_init=10,
        max_iter=300, tol=0.0001, verbose=0, random_state=None,
        copy_x=True, algorithm='auto')

  #convertion np.darray en pd.df
  x2 = pd.DataFrame(x)

  #nombres de fragments par classe et centres de chaque classe stockés dans df
  classes = modele_km.labels_

  count = pd.DataFrame(np.unique(classes,return_counts=True)[1],
                      columns=["Nombres de fragments"])
  centres = pd.DataFrame(modele_km.cluster_centers_, columns=x2.columns)

  #graphique des classes avec 
  plt.figure(figsize=(12,7))
  markers = ["+", "s", "^", "v", ">"]
  for val, mark in zip(np.unique(classes), markers):
    plt.scatter(x2[0][classes==val],
                x2[1][classes==val],marker=mark,
                label="classe% i"%(val+1))
  plt.title("Classes des fragments de taille 5")
  plt.legend()

#plot_inertia(x)

#plot_classes(x)

#afficher informations
#pd.set_option('precision',2)
#print(pd.concat([count, pd.DataFrame(centres,columns=x2.columns)],
#                axis =1).T.head())
#pd.reset_option('precision')