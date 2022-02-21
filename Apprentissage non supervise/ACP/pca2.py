# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 19:15:51 2020

@author: lucho
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits


digit = load_digits()
digit.keys()
digit.target_names
digit.data.shape
plt.gray()
plt.matshow(digit.images[0])
plt.show()

model =PCA(n_components =2)
model.fit(digit.data)

data_reduced = model.fit_transform(digit.data)
plt.scatter(data_reduced[:,0],data_reduced[:,1])

plt.scatter(data_reduced[:,0],data_reduced[:,1], c=digit.target)
plt.colorbar()
plt.show()

#composantes principales
model.components_
model.components_[0,:]
model.components_[1,:]

# Reduction de dimension
model = PCA(n_components =64)
model.fit(digit.data)
model.explained_variance_ratio_
np.cumsum(model.explained_variance_ratio_)
100*np.cumsum(model.explained_variance_ratio_)
plt.plot(np.cumsum(model.explained_variance_ratio_))
np.argmax(100*np.cumsum(model.explained_variance_ratio_)>99)

########
model = PCA(n_components =28)
data_reduced = model.fit_transform(digit.data)
data_recovered = model.inverse_transform(data_reduced )
plt.imshow(data_recovered[9].reshape((8,8)))
####

#Faire varier le n_components 1,2,3,4, etc ....

### Derniere technique
model = PCA(n_components =0.95)
data_reduced = model.fit_transform(digit.data)
data_recovered = model.inverse_transform(data_reduced )
plt.imshow(data_recovered[0].reshape((8,8)))
model.n_components_
