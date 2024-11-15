import math
from collections import Counter

import numpy as np
import pandas as pd
import pyideem
import random

import qiskit
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps

qasm_file = "Task3/ftest_2.qasm"

data = pd.read_csv('Task3/task-3-dataset.csv')
texts = data.iloc[:, 0].tolist()
y_labels = data.iloc[:, 1].replace({'+': 1, '-': 0}).tolist()

data_test = pd.read_csv('Task3/test50.csv')
texts_test = data_test.iloc[:, 0].tolist()
y_labels_test = data_test.iloc[:, 1].replace({'+': 1, '-': 0}).tolist()


def vectorize_text(texts, vector_size=20):
    all_words = ' '.join(texts).split()
    word_counts = Counter(all_words)
    common_words = [word for word, _ in word_counts.most_common(vector_size)]

    short_vectors = []
    for text in texts:
        word_set = set(text.split())
        vector = [1 if word in word_set else 0 for word in common_words]
        short_vectors.append(np.array(vector))

    return np.array(short_vectors)

X_data_vect = vectorize_text(texts)
X_test_vect = vectorize_text(texts_test)


def create_qasm(params):
    qc = QuantumCircuit(3, 3)

    qc.x(0)
    qc.x(1)
    qc.x(2)
    qc.reset(0)
    qc.reset(1)
    qc.reset(2)

    qc.p(params[0], 0)
    qc.p(params[1], 1)
    qc.p(params[2], 2)

    qc.cx(0, 1)
    qc.cx(1, 2)

    qc.p(params[3], 0)
    qc.p(params[4], 1)
    qc.p(params[5], 2)

    qc.h(0)
    qc.h(1)
    qc.h(2)

    qc.measure([0, 1, 2], [0, 1, 2])

    qasm_code = dumps(qc)

    with open(qasm_file, "w") as file:
        file.write(qasm_code)


def train_test_split_custom(X_train, y_train, X_test, y_test):
    data_size = len(X_train)
    test_size = len(X_test)
    indices = list(range(data_size))
    random.shuffle(indices)

    print("data_size ", data_size)
    print("test_size ", test_size)
    print("indices ", indices)

    X_train = [X_train[i] for i in indices]
    X_test = [X_test[i] for i in X_test]
    y_train = [y_train[i] for i in indices]
    y_test = [y_test[i] for i in y_test]

    print("X_train ", X_train)
    print("X_test ", X_test)

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


X_train, X_test, y_train, y_test = train_test_split_custom(X_data_vect, y_labels, X_test_vect, y_labels_test)

qc = pyideem.QuantumCircuit.loadQASMFile(str(qasm_file))

backend = pyideem.StateVector(3)

def combine_qubits_via_xor(counts):
    most_frequent = max(counts, key=counts.get)
    bits = [int(bit) for bit in most_frequent]
    result = bits[0] ^ bits[1] ^ bits[2]
    return result


def quantum_classifier(inputs, qc, backend, shots=100):
    results = []

    for input_vector in inputs:
        input_vector = np.array(input_vector).reshape(-1)

        params = [3.14 / 2 * (input_vector[i] ^ input_vector[2 * i] ^ input_vector[3 * i]) for i in range(6)]

        create_qasm(params)

        result = qc.execute(shots, backend, noise_cfg=None, return_memory=True)

        counts = result.counts
        result_bit = combine_qubits_via_xor(counts)
        results.append(result_bit)

    return np.array(results)


train_predictions = quantum_classifier(X_train, qc, backend)
test_predictions = quantum_classifier(X_test, qc, backend)


def accuracy_score_custom(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


accuracy = accuracy_score_custom(y_test, test_predictions)
print("Accuracy:", accuracy)
