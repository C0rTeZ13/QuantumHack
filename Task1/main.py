import math
from time import time

import numpy as np

from scipy.sparse import coo_matrix

from scipy.sparse import random

from scipy import stats, sparse

import pyqiopt as pq

# Данные
data = np.genfromtxt("Task1/task-1-stocks.csv", delimiter=',')
cost_coeffs = data[0]
profit_coeffs = data[99] - data[0]

max_risk = 0.2
budget = 1_000_000
lambda_budget = 0.2
mu_risk = 0.5
deviation_risk = 0.6

num_stocks = 100
num_bits = 10

# Доходности
returns = np.diff(data, axis=1)
# Ковариация
covar = np.cov(returns)

# Инициализация QUBO
Q = np.zeros((num_stocks * num_bits, num_stocks * num_bits))

# Целевая функция: максимизация прибыли
for i in range(num_stocks):
    for k in range(num_bits):
        Q[i * num_bits + k, i * num_bits + k] -= profit_coeffs[i] * (2**k)

# Ограничение бюджета
for i in range(num_stocks):
    for k in range(num_bits):
        for j in range(num_stocks):
            for l in range(num_bits):
                Q[i * num_bits + k, j * num_bits + l] += (
                    lambda_budget * cost_coeffs[i] * cost_coeffs[j] * (2**k) * (2**l)
                )
        Q[i * num_bits + k, i * num_bits + k] -= 2 * lambda_budget * budget * cost_coeffs[i] * (2**k)

# Ограничение по риску
for i in range(num_stocks):
    for j in range(num_stocks):
        for k in range(num_bits):
            for l in range(num_bits):
                Q[i * num_bits + k, j * num_bits + l] += (
                    mu_risk * covar[i][j] * (2**k) * (2**l)
                )
                Q[i * num_bits + k, j * num_bits + l] += (
                    deviation_risk * (covar[i][j] - max_risk)**2 * (2**k) * (2**l)
                )

arr_sp = sparse.coo_matrix(Q)
print("QUBO Matrix:")
print(arr_sp)

start = time()
print("Sampling")
sol = pq.solve(arr_sp, number_of_runs=1, number_of_steps=500, return_samples=True, verbose=10)
samples = sol.samples
print("Samples:")
print(samples)

print("Script time:", time() - start)

Q = sparse.triu(Q)
Q = Q.toarray()

portfel = []

for key, sample in samples.items():
    portfel_sample = []
    binary_array = sample[0]

    for i in range(0, len(binary_array), 10):
        binary_str = ''.join(map(str, binary_array[i:i+10]))
        decimal_value = int(binary_str, 2)
        portfel_sample.append(decimal_value)
        
    portfel.append(portfel_sample)

print("Портфель:", portfel)

# Проверка
number = 0
good = []
good_returns = []
for var in portfel:
    number += 1
    sum_ = [0] * num_stocks
    returns_ = [0] * (num_stocks - 1)

    # Сумма в портфеле на каждый день
    for i in range(num_stocks):
        for j in range(num_stocks):
            sum_[i] += var[j] * data[i][j]

    # Доходности
    for i in range(num_stocks - 1):
        returns_[i] = (sum_[i+1] - sum_[i]) / sum_[i]

    # Средняя доходность
    median_returns = 0
    for i in range (num_stocks - 1):
        median_returns += returns_[i]
    median_returns = median_returns / (num_stocks - 1)
        
    # Уровень риска
    risk = 0
    for i in range(num_stocks-1):
        risk += ((returns_[i] - median_returns) ** 2) / (num_stocks - 2)
    risk = math.sqrt((num_stocks-1) * risk)

    print(f"For VAR {number}")
    print(f"Risk: {risk}")
    print(f"Budget START: {sum_[0]}")
    print(f"Budget END: {sum_[-1]}")

    if budget*0.95 < sum_[0] <= budget and max_risk - 0.05 < risk < max_risk:
        good.append(number)
        good_returns.append(returns_)
print(f"GOOD: {good}")
np.save("Task1/good_returns.npy", good_returns)