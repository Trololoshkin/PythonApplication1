#1 объявление переменных, и всех import
#2 функции ордеров (покупка, продажа, ждать, закывать)
#3 интерфейс, два поля для ввода информации
# 3.1текущий баланс
# 3.2 процент - будет использован в фукциях ордеров
#4 три кнопки
# 4.1 первая - загрузка файла , на которм будет проводится обчуение Нейросети
# 4.2 вторая -  подключение базы нейросети (создание, если база отсутствует). Используем SQLite.
# 4.3 третья — запуск обучения нейросети
#5 функция обучения нейросети на основе фала из пукта 4.1, и сохранение базы нейронной сети в файле указанном в пункте 4.2.
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

class Order():
    def __init__(self, balance, percent):
        self.balance = balance
        self.percent = percent
        
    def buy(self, price):
        amount = self.balance * self.percent
        self.balance -= amount
        return amount/price
        
    def sell(self, amount, price):
        self.balance += amount * price
        return self.balance
        
    def wait(self):
        return None
        
    def close(self, amount, price):
        self.balance += amount * price
        return self.balance

def load_csv_file():
    global df
    # Выбор файла
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    # Чтение файла
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # Создание/подключение базы данных
    db_file = os.path.splitext(file_path)[0] + ".db"
    conn = sqlite3.connect(db_file)
    # Проверка наличия таблицы в базе данных
    c = conn.cursor()
    c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='prices'")
    if c.fetchone()[0] != 1:
        # Создание таблицы, если она не существует
        c.execute('''CREATE TABLE prices
                     (date TEXT PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, volume REAL)''')
        # Запись данных из DataFrame в таблицу
        for index, row in df.iterrows():
            c.execute("INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?)", (index, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]))
        conn.commit()
    else:
        # Чтение данных из таблицы в DataFrame
        df = pd.read_sql_query("SELECT * FROM prices", conn, index_col="date", parse_dates=["date"])
    # Закрытие соединения с базой данных
    conn.close()

 

def main():
    root = tk.Tk()
    root.geometry('200x400')
    root.title('Trading Bot')
    
    # Fields for input
    balance_field = tk.Entry(root)
    balance_field.pack()
    
    percent_field = tk.Entry(root)
    percent_field.pack()
    
    # Buttons
    def open_file():
        file_path = filedialog.askopenfilename()
        print('Selected file:', file_path)
    
    load_file_button = tk.Button(root, text='Load CSV file', command=open_file)
    load_file_button.pack()
    
    def open_database():
        db_file = filedialog.askopenfilename()
        print('Selected database:', db_file)
    
    load_database_button = tk.Button(root, text='Load Neural Network', command=open_database)
    load_database_button.pack()
    
    def train_network(database_path, status_label):
    # проверяем существует ли база данных
    if not os.path.isfile(database_path):
        conn = sqlite3.connect(database_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE stocks
                    (date text, open real, high real, low real, close real, volume real)''')
        conn.commit()
        conn.close()

    try:
        # загружаем csv файл
        csv_file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV Files", "*.csv")])
        if not csv_file_path:
            return
        status_label.config(text="Reading CSV file...")

        # проверяем наличие заголовков
        with open(csv_file_path, newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)
            if not all(header in headers for header in ['Open', 'High', 'Low', 'Close']):
                status_label.config(text="CSV file is missing required columns")
                return

        # загружаем данные из csv файла
        df = pd.read_csv(csv_file_path)

        # создаём нейронную сеть
        input_size = 4
        hidden_size = 16
        output_size = 1
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

        # определяем функцию потерь и оптимизатор
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        # генерируем случайные индексы для тренировочного набора данных
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        train_indices = indices[:int(0.7*len(df))]

        # тренируем модель
        status_label.config(text="Training network...")
        for i in range(200):
            total_loss = 0
            for j in train_indices:
                input_tensor = torch.tensor(df.iloc[j][['Open', 'High', 'Low', 'Close']].values)
                output_tensor = torch.tensor(df.iloc[j]['Close'])
                optimizer.zero_grad()
                loss = criterion(model(input_tensor), output_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_indices)
            if i % 10 == 0:
                status_label.config(text="Training network... Epoch {}, Avg. Loss: {:.4f}".format(i, avg_loss))
        status_label.config(text="Network trained successfully")

        # сохраняем модель в базу данных
        conn = sqlite3.connect(database_path)
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS model")
        c.execute("CREATE TABLE model (params BLOB)")
        params = model.state_dict()
        c.execute("INSERT INTO model (params) VALUES (?)", (pickle.dumps(params),))
        conn.commit()
        conn.close()

    except Exception as e:
        print(e)
        status_label.config(text="Failed to train network")
    
    train_button = tk.Button(root, text='Train Network', command=train_network(database_path, status_label))
    train_button.pack()
    
    root.mainloop()

if __name__ == '__main__':
    main()
