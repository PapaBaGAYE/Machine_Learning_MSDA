# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 01:53:03 2021

@author: lucho
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn import datasets

#chargement de base de données iris
iris = datasets.load_iris()

###familiariser avec les données
print(iris)
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

df  = pd.DataFrame({
        'x' : iris.data[:,0],
        'y' : iris.data[:,1],
        'cluster' : iris.target
    })



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

#Cluster K-means
model=KMeans(n_clusters=3)
#ajuster le modèle de données
model.fit(x)
print(model.labels_)
#plt.scatter(x.Petal_Length, x.Petal_width)

colortable =np.array(['yellow','green','blue'])
plt.scatter(x.Petal_Length, x.Petal_width,c=colortable[iris.target],s=40)

plt.scatter(x.Petal_Length, x.Petal_width,c=colortable[model.labels_],s=40)
plt.show()
sm.accuracy_score(iris.target,model.predict(x))
sm.silhouette_score(iris.data,model.labels_)
