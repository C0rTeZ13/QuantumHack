import numpy as np
import matplotlib.pyplot as plt

y_values = np.load("good_returns2.npy")
x_values = np.arange(1, len(y_values[0]) + 1)

# Построение графика
plt.figure(figsize=(10, 6))  # Опционально: установка размера графика
plt.plot(x_values, y_values[0], marker='o', linestyle='-', color='b')  # Построение графика
plt.xlabel('Days')  # Название оси X
plt.ylabel('Returns')  # Название оси Y
plt.title('График доходности стратегии')  # Заголовок графика
plt.grid(True)  # Сетка для удобства чтения
plt.savefig('returns2.png')