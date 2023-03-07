import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sqlite3
import time
import csv
import os


# Окошко с кнопошками
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

# Функции выставления ордеров на покупку и продажу
def buy(price, capital):
    units = capital // price
    return units

def sell(price, units):
    return units * price

# Функция для закрытия сделки и добавления прибыли/убытка в исходный капитал
def close_trade(trade, price, capital):
    if trade['type'] == 'buy':
        profit = sell(price, trade['units']) - trade['cost']
    else:
        profit = trade['cost'] - sell(price, trade['units'])
    return capital + profit

# Загрузка данных из файла
df = pd.read_csv('quotes.csv')

# Изменение столбца "Изменение" на числовое значение
df['Изменение'] = df['Изменение'].str.replace('%', '').astype(float)

# Выбор колонок с ценой и изменением для обучения модели
train_df = df[['Цена', 'Изменение']][:-10]

# Выбор колонок с ценой и изменением для тестирования модели
test_df = df[['Цена', 'Изменение']][-10:]

# Создание экземпляра классификатора и обучение модели
model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000)
model.fit(train_df, np.array(df['Действие'][:-10]))

# Тестирование модели
capital = 1000
current_trade = None
for index, row in test_df.iterrows():
    # Вычисляем текущее значение цены и изменения
    price = row['Цена']
    change = row['Изменение']

    # Делаем предсказание
    X_test = np.array([[price, change]])
    prediction = model.predict(X_test)[0]

    # Проверяем, нужно ли открыть новую сделку
    if prediction == 'buy' and not current_trade:
        units = buy(price, capital)
        cost = price * units
        current_trade = {'type': 'buy', 'units': units, 'cost': cost}
        capital -= cost
    elif prediction == 'sell' and current_trade:
        capital = close_trade(current_trade, price, capital)
        current_trade = None

# Вывод итогового капитала
print(f"Final capital: {capital}")

# Функция для создания таблицы нейросети


# функция для создания таблицы в базе данных
def create_table(cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS stock_prices
                    (id INTEGER PRIMARY KEY,
                     date TEXT,
                     open REAL,
                     high REAL,
                     low REAL,
                     close REAL)''')

# функция для загрузки файла csv
def load_csv_file():
    file_path = filedialog.askopenfilename()
    return file_path

# функция для создания/подключения к базе данных
def create_database():
    conn = sqlite3.connect("neural_net.db")
    cursor = conn.cursor()
    create_table(cursor)
    return conn, cursor

# функция для обучения нейронной сети
def train_neural_net():
    # код обучения нейронной сети
    pass

# создание окна
root = tk.Tk()
root.geometry("200x400")
root.title("Stock Prediction")

# добавление элементов на окно
balance_label = tk.Label(root, text="Balance:")
balance_label.pack()

balance_entry = tk.Entry(root)
balance_entry.pack()

bet_label = tk.Label(root, text="Bet (% of balance):")
bet_label.pack()

bet_entry = tk.Entry(root)
bet_entry.pack()

load_file_button = tk.Button(root, text="Load File", command=load_csv_file)
load_file_button.pack()

create_db_button = tk.Button(root, text="Create/Connect to Database", command=create_database)
create_db_button.pack()

train_button = tk.Button(root, text="Train Neural Network", command=train_neural_net)
train_button.pack()

# запуск окна
root.mainloop()

