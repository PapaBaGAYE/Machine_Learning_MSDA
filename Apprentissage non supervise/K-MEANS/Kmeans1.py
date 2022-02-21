# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 23:37:47 2020

@author: lucho
"""

#importation des librairies nécessaires

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from utilitaires import  find_permutation


#Générer les données

dataset = make_blobs(n_samples=500,
                     centers=3,
                     n_features=2,
                     cluster_std=0.7,
                     random_state=0
                     )
# variables explicatives (p=2)
data =dataset[0]
#labels
target = dataset[1]

#Affichage des données
plt.scatter(data[:,0],data[:,1], s=40)
plt.show()

#Créer un modèle kmeans
model = KMeans(n_clusters=3)
#Ajuster le modèle aux données
model.fit(data)
#prediction
y_kmeans = model.predict(data) # même chose que model.labels_
model.labels_ #les labels prédits
model.cluster_centers_ # centroides
model.inertia_ #inertie
model.score(data) #-inertie

###### PROBLEME LES LABELS SONT PERMUTES  on va vérifier çà avec les graphiques
# et le score
# Representation graphique 1
plt.scatter(data[model.labels_==0][:,0],
            data[model.labels_==0][:,1],s=50,color ='BLUE')
plt.scatter(data[model.labels_==1][:,0],
            data[model.labels_==1][:,1],s=50,color ='GREEN')
plt.scatter(data[model.labels_==2][:,0],
            data[model.labels_==2][:,1],s=50,color ='CYAN')
plt.scatter(model.cluster_centers_[:,0],
            model.cluster_centers_[:,1],c='r',marker='*')
plt.show()

# Representation graphique  faite d'une autre manière
plt.scatter(data[:,0],data[:,1],c=model.labels_)
plt.scatter(model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1], c='RED', s=200, alpha=0.5)


#EVALUATION  
sm.accuracy_score(target, model.labels_,normalize=True)
#conclusion  : score peut être faible puisque les labels  des classes 
#peuvent être  permutés

# RESOLUTION DU PROBLEME

# Representation graphique 2
permutation = find_permutation(3, target, model.labels_)
new_labels = [ permutation[label] for label in model.labels_]

plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

#EVALUATION
sm.accuracy_score(target,new_labels,normalize=True)
sm.silhouette_score(data,model.labels_)

# extraire les données des clusters
data[ model.labels_==0] # premier cluster 
data[model.labels_ ==1] # Second cluster 
data[model.labels_ ==2] #troisieme cluster
#Choix du K
#ELBOW METHOD : Détecter une zone de coude dans la minimisation
# du Coût (inertia_)
inertia = []
K_range = range(1,20)
for k in K_range:
    model = KMeans(n_clusters = k).fit(data)
    inertia.append(model.inertia_)

plt.plot(K_range,inertia)
plt.xlabel('nombre de clusters')
plt.ylabel('Cout du modele(inertia')




