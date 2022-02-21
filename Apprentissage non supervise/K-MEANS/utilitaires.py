# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:33:24 2021

@author: lucho
"""
import scipy



def find_permutation(n_clusters, real_labels, labels):
    permutation=[]
    for i in range(n_clusters):
        idx = labels == i
        new_label=scipy.stats.mode(real_labels[idx])[0][0]
        permutation.append(new_label)
    return permutation
