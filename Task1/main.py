from time import time

import numpy as np

from scipy.sparse import coo_matrix

from scipy.sparse import random

from scipy import stats

data = np.genfromtxt("task-1-stocks.csv", delimiter=',')

output_data = np.ones((100, 100))
risk_coef = 2

profit = np.ones((100, 100))
for i in range(99):
    profit[i] = (data[i + 1] - data[i]) / data[i]

covar = np.ones((100, 100))

for i in range(100):
    for j in range(100):
        covar[i][j] = np.cov(profit[i], profit[j])[0, 1]

print(covar)

for i in range(100):
    for j in range(100):
        output_data[i][j] = -((data[i][99] - data[i][0]) / data[i][0]) + risk_coef * covar[i][j]

print(output_data)
