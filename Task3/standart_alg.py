# Импортирование библиотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Создание данных
data = pd.read_csv('task-3-dataset.csv')

# Разделение данных на признаки и метки
X = data.iloc[:, 0]  # Первый столбец
y = data.iloc[:, 1]  # Второй столбец

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование всех данных в строковый формат
X_train = X_train.astype(str)
X_test = X_test.astype(str)

# Преобразование текста в векторное представление с использованием TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Создание и обучение модели
model = LogisticRegression(random_state=42)

# Обучаем модель на векторизованных данных
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
