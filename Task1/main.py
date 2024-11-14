from time import time

import numpy as np

from scipy.sparse import coo_matrix

from scipy.sparse import random

from scipy import stats

import pyqiopt as pq

risk_level = 0.2
data = np.load("task-1-stocks.csv")
P = 1000000

for i in data:

density = 0.01

random_state = 42

sparse_matrix = random(rows, cols, density=density, random_state=random_state)

arr = sparse_matrix.todense()

arr = (arr + arr.T) / 2

print("Non zero elements:", np.count_nonzero(arr))

start = time()

print("Numpy matrix example")

sol = pq.solve(arr, number_of_runs=1, number_of_steps=100, return_samples=False, verbose=10)

print(sol.vector, sol.objective)

arr_sp = coo_matrix(arr) # for pyqiopt input use COO format only

print("Sparse COO matrix example")

sol = pq.solve(arr_sp, number_of_runs=1, number_of_steps=100, return_samples=False, verbose=10)

print(sol.vector, sol.objective)

print("Sampling example")

sol = pq.solve(arr_sp, number_of_runs=1, number_of_steps=100, return_samples=True, verbose=10)

print(sol.samples)

print("Script time:", time()-start)