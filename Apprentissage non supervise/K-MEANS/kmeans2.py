# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 00:31:11 2020

@author: lucho
"""

import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from utilitaires import  find_permutation




#chargement de base de données iris
iris = datasets.load_iris()

print(iris)
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)



#Stocker les données en tant que DataFrame Pandas
x=pd.DataFrame(iris.data)
# définir les noms de colonnes
x.columns=['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
y=pd.DataFrame(iris.target)
y.columns=['Targets']


# Representation 3D des données (les données ont 4 dimensions 
#ici je choisis les 3 dernières dimensions)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x.Sepal_width, x.Petal_Length, x.Petal_width)
plt.title("Données initiales")
plt.show()

#Representation des donnees en 2D : je travaille avec les deux dernières 
#dimensions.
colormap =np.array(['BLUE','GREEN','CYAN'])
plt.scatter(x.Petal_Length, x.Petal_width,s=40)

#Cluster K-means
model=KMeans(n_clusters=3)
#ajuster le modèle de données (on ne travaille qu'avec les deux dernières dimensions)
model.fit(x)
print(model.labels_)
print(model.cluster_centers_)
#Affichage des clusters
dat = iris.data[:,2:4] # donnees ne contenant que 'Petal_Length','Petal_width'
plt.scatter(dat[model.labels_==0][:,0], dat[model.labels_==0][:,1],
            c='BLUE',s=40)
plt.scatter(dat[model.labels_==1][:,0], dat[model.labels_==1][:,1],
            c='GREEN',s=40)
plt.scatter(dat[model.labels_==2][:,0], dat[model.labels_==2][:,1],
            c='CYAN',s=40)
plt.show()

#Autre affichage des ces clusters
permutation = find_permutation(3, iris.target, model.labels_)
new_labels = [ permutation[label] for label in model.labels_]
plt.scatter(x.Petal_Length, x.Petal_width, c=colormap[new_labels], s=50, cmap='viridis')
centers = model.cluster_centers_


sm.accuracy_score(iris.target,new_labels)
sm.silhouette_score(iris.data,model.labels_)

inertia = []
K_range = range(1,20)
for k in K_range:
    model = KMeans(n_clusters = k).fit(x)
    inertia.append(model.inertia_)

plt.plot(K_range,inertia)
plt.xlabel('nombre de clusters')
plt.ylabel('Cout du modele(inertia')
