import pyideem
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Загрузка и подготовка данных
data = pd.read_csv('task-3-dataset.csv')  # Замените на свой путь к файлу
texts = data.iloc[:, 0].tolist()
y_labels = data.iloc[:, 1].replace({'+': 1, '-': 0}).tolist()

# 2. Векторизация текста с использованием TF-IDF
vectorizer = TfidfVectorizer(max_features=10)  # Можно увеличить количество признаков
X_vectorized = vectorizer.fit_transform(texts).toarray()

# Разбиение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_labels, test_size=0.2)

# 3. Инициализация квантового устройства
qasm_file = "/workspace/ftest_1.qasm"  # Укажите путь к вашему QASM файлу
qc = pyideem.QuantumCircuit.loadQASMFile(str(qasm_file))

# Инициализация 3-кубитного состояния
backend = pyideem.StateVector(3)

# 4. Определение функции для классификации с использованием квантового алгоритма
def quantum_classifier(inputs, qc, backend, shots=10):
    results = []

    for input_vector in inputs:
        # Модификация квантового состояния в зависимости от признаков
        # Здесь можно использовать различные кодировки признаков
        input_vector = np.array(input_vector).reshape(-1)

        # Выполнение квантового алгоритма
        result = qc.execute(shots, backend, noise_cfg=None, return_memory=True)

        # Получение результата
        counts = result.counts
        result_bit = max(counts, key=counts.get)  # Получаем наиболее вероятный битовый результат
        results.append(int(result_bit, 2))  # Преобразуем в целое число

    return np.array(results)


# 5. Обучение модели и предсказания
# Используем квантовый классификатор для предсказания
train_predictions = quantum_classifier(X_train, qc, backend)
test_predictions = quantum_classifier(X_test, qc, backend)

# 6. Оценка точности модели
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy:", accuracy)
