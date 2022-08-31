import mdtraj as md
from mdtraj import *
import numpy as np
from numpy import *
import sys
import matplotlib.pyplot as plt
import itertools
import os


def all_Calpha_contact_maps(file, parent_dir):
  name=str(file)
  name = name.replace('.pdb', '')
  print("Fichier :\n",file)
  #Créer un dossier pour chaque pdb
  directory = name
  path = os.path.join(parent_dir, directory)
  os.mkdir(path)

  #Chargement du fichier avec mdtraj
  traj = md.load(file)
  topology = traj.topology
  
  #print('MD imformation: %s\n' % topology)
  #print('All residues: %s\n' % [residue for residue in traj.topology.residues])
  #print('All atoms: %s\n' % [atom for atom in traj.topology.atoms])


  #Cartes de contacts
  size_frag=[3, 5, 7, 9, 11, 13]
    ##pas de +1 pour compenser 0 car range() exclut le dernier
  i0 = 0

  traj = md.load(file)
  contact_ca_all = md.compute_contacts(traj, contacts="all", scheme='ca', ignore_nonprotein=True )
  square_all = md.geometry.squareform(contact_ca_all[0], contact_ca_all[1])

  for j0 in size_frag :
    list_groups = []
    
    #i pour la totalité des C alpha
    for i in range(0,len(square_all[0])-1-j0):
    #for i in range(0, 1):
      group = [i for i in range(i0+i,j0+i)]
      list_groups.append(group)

    #Calculs des distances pour chaque paire
    for group in list_groups:
      pairs = list(itertools.product(group, group))

      contact_ca = md.compute_contacts(traj, pairs, scheme='ca', ignore_nonprotein=True )
      distances = contact_ca[0]
      res_pairs = contact_ca[1]

      square = md.geometry.squareform(distances, res_pairs)
      sq_arr=square[0]

      #Supprimer les colonnes et lignes de zéros
      idx_row = np.argwhere(np.all(sq_arr[:, ...] == 0, axis=1))
      #print("row\n",idx_row)
      idx_col = np.argwhere(np.all(sq_arr[...,:] == 0, axis=0))
      #print("col\n",idx_col)
      arr2 = np.delete(sq_arr, idx_row, axis=0)
      arr3 = np.delete(arr2, idx_col, axis=1)

      #Générer les images
      plt.imshow(arr3, cmap="Greys_r")
      plt.axis('off')
      plt.savefig("{0}/{1}_size{2}_idx{3}.png".format(path, name,j0, list_groups.index(group)), 
                  format="png", bbox_inches='tight',transparent=True, pad_inches=0)
      plt.close()
