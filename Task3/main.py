import numpy as np
import pandas as pd
import pyideem
import random

# 1. Загрузка и подготовка данных
data = pd.read_csv('Task3/task-3-dataset.csv')  # Замените на свой путь к файлу
texts = data.iloc[:, 0].tolist()
y_labels = data.iloc[:, 1].replace({'+': 1, '-': 0}).tolist()


# 2. Векторизация текста (простое представление текста)
def vectorize_text(texts):
    vectorized = []
    for text in texts:
        vectorized.append(np.array([ord(char) for char in text]).sum())
    return np.array(vectorized)

X_vectorized = vectorize_text(texts)


# 3. Разбиение данных на обучающий и тестовый наборы вручную
def train_test_split_custom(X, y, test_size=0.2):
    data_size = len(X)
    test_size = int(data_size * test_size)
    indices = list(range(data_size))
    random.shuffle(indices)

    X_train = [X[i] for i in indices[test_size:]]
    X_test = [X[i] for i in indices[:test_size]]
    y_train = [y[i] for i in indices[test_size:]]
    y_test = [y[i] for i in indices[:test_size]]

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


X_train, X_test, y_train, y_test = train_test_split_custom(X_vectorized, y_labels)

# 4. Инициализация квантового устройства
qasm_file = "Task3/ftest_1.qasm"  # Укажите путь к вашему QASM файлу
qc = pyideem.QuantumCircuit.loadQASMFile(str(qasm_file))

# Инициализация 3-кубитного состояния
backend = pyideem.StateVector(3)


# 5. Определение функции для классификации с использованием квантового алгоритма
def quantum_classifier(inputs, qc, backend, shots=100):
    results = []

    for input_vector in inputs:
        # Модификация квантового состояния в зависимости от признаков
        input_vector = np.array(input_vector).reshape(-1)

        theta = (input_vector.sum() % 360) * (np.pi / 180)  # Преобразуем в радианы
        phi = ((input_vector.sum() * 2) % 360) * (np.pi / 180)  # Преобразуем в радианы

        # Выполнение квантового алгоритма
        result = qc.execute(shots, backend, noise_cfg=None, return_memory=True)

        # Получение результата
        counts = result.counts
        result_bit = max(counts, key=counts.get)  # Получаем наиболее вероятный битовый результат
        results.append(int(result_bit, 2))  # Преобразуем в целое число

    return np.array(results)


# 6. Обучение модели и предсказания
train_predictions = quantum_classifier(X_train, qc, backend)
test_predictions = quantum_classifier(X_test, qc, backend)


# 7. Оценка точности модели
def accuracy_score_custom(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


accuracy = accuracy_score_custom(y_test, test_predictions)
print("Accuracy:", accuracy)
