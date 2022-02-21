# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:36:06 2020

@author: lucho
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import sklearn.metrics as sm
#from sklearn.metrics import *
#Generation de donnees
colormap=np.array(['Yellow','green','blue'])
X,y = make_blobs(n_samples =100, centers = 3,cluster_std=0.60,random_state=0)
X = X[:, ::-1]
plt.scatter(X[:,0], X[:,1],c=colormap[y])

model = KMeans(n_clusters = 3,random_state=0)
model.fit(X)
model.labels_
model.predict(X)

plt.scatter(X[:,0],X[:,1],c=colormap[model.predict(X)],s=40)
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],c='r')
plt.show()
sm.accuracy_score(y,model.predict(X))

model.cluster_centers_
model.inertia_
model.score(X)
#silhouette_score(model)

#Choix du K
#ELBOW METHOD : Détecter une zone de coude dans la minimisation
# du Coût (inertia_)
inertia = []
K_range = range(1,20)
for k in K_range:
    model = KMeans(n_clusters = k).fit(X)
    inertia.append(model.inertia_)

plt.plot(K_range,inertia)
plt.xlabel('nombre de clusters')
plt.ylabel('Cout du modele(inertia')
