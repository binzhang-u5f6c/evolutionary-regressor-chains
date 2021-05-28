# -*- coding: utf-8 -*-
"""Generate synthetic data.

Generate synthetic data.
"""
from copy import deepcopy as dcp

import numpy as np
import pandas as pd

# configure vars
syndata_id = 4
n, m, d = 50000, 6, 4
G = np.zeros((m, m))
single_chain = np.random.permutation(m)
for i in range(1, m//2):
    G[single_chain[i-1], single_chain[m//2]] = np.random.uniform(-3, 3)
for i in range(m//2, m):
    G[single_chain[m//2-1], single_chain[i]] = np.random.uniform(-3, 3)
coef = np.random.uniform(-3, 3, size=(m, d))
intercept = np.random.uniform(-3, 3, size=m)


def topological_sort(G):
    """Yield a topological order of a graph."""
    m = len(G)
    unsorted_v = [i for i in range(m)]
    sorted_v = []
    while len(unsorted_v) > 0:
        temp_v = []
        for v in unsorted_v:
            flag = 0
            for i in range(m):
                if G[i, v] != 0:
                    flag = 1
            if flag == 0:
                temp_v.append(v)
        for v in temp_v:
            unsorted_v.remove(v)
            sorted_v.append(v)
            for i in range(m):
                G[v, i] = 0
    return sorted_v


x = np.random.normal(size=(n, m, d))
y = np.zeros((n, m))
for i, coefi in enumerate(coef):
    y[:, i] = np.dot(x[:, i, :], coefi) + \
        intercept[i] * np.ones(n)
topological_order = topological_sort(dcp(G))
for i in topological_order:
    for j in range(m):
        y[:, i] += G[j, i] * y[:, j]

xy = np.hstack((x.reshape(-1, d), y.reshape(-1, 1)))
cols = ['feature_{}'.format(i+1) for i in range(d)]
cols.append('label')
df = pd.DataFrame(xy, columns=cols)
df.to_csv('data/multistream.synthetic{}.csv'.format(syndata_id), index=False)

# print info
np.set_printoptions(precision=4)
print('################ Synthetic dataset '
      '#{} ################'.format(syndata_id))
print('n = {}, m = {}, d = {}'.format(n, m, d))
print('Coef:')
print(coef)
print('Intercept:')
print(intercept)
print('Correlations:')
print(G)
print()
