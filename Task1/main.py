from time import time

import numpy as np

from scipy.sparse import coo_matrix

from scipy.sparse import random

from scipy import stats

data = np.genfromtxt("task-1-stocks.csv", delimiter=',')

output_data = np.ones((100, 100))
risk_coef = 2

profit = np.ones((100, 100))
for i in range(100):
    for j in range(99):
        profit[i][j] = (data[i][j + 1] - data[i][j]) / data[i][j]

covar = np.ones((100, 100))

for i in range(100):
    for j in range(100):
        covar[i][j] = np.cov(profit[i], profit[j])[0, 1]

print(covar)

for i in range(100):
    for j in range(100):
        output_data[i][j] = round(-((data[j][99] - data[j][0]) / data[j][0]) + risk_coef * covar[i][j])

for i in range(100):
    for j in range(100):
        print(output_data[i, j], end=' ')
    print(' ')