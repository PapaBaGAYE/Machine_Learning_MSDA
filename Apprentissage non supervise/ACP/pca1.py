# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:35:56 2020

@author: lucho
"""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

#generation de donnees 3D
data = np.random.randn(500,3)
data.shape
# Affichage du nuage de points
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2])
plt.title("Données initiales")
plt.show()


#ACP DES DONNEES
pca = PCA(n_components=3)
pca.fit(data)

print("Pourcentage de variance expliquée : ")
print(pca.explained_variance_ratio_)
np.cumsum(pca.explained_variance_ratio_)
print("Composantes principales : ")
print(pca.components_)

data_reduced = pca.fit_transform(data)
plt.scatter(data_reduced[:,0],data_reduced[:,1])

pca = PCA(n_components=2)
pca.fit(data)

print("Pourcentage de variance expliquée : ")
print(pca.explained_variance_ratio_)
np.cumsum(pca.explained_variance_ratio_)
print("Composantes principales : ")
print(pca.components_)

data_reduced = pca.fit_transform(data)
plt.scatter(data_reduced[:,0],data_reduced[:,1])
