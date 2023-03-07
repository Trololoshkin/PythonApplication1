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
        try:
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT price, next_price FROM stock_data")
            rows = cursor.fetchall()

            # Convert the data into numpy arrays
            dataset = np.array(rows)
            x_train = dataset[:, 0].reshape(-1, 1).astype('float32')
            y_train = dataset[:, 1].reshape(-1, 1).astype('float32')

            # Normalize the data
            x_norm = (x_train - np.mean(x_train)) / np.std(x_train)
            y_norm = (y_train - np.mean(y_train)) / np.std(y_train)

            # Convert the numpy arrays to PyTorch tensors
            x_tensor = torch.from_numpy(x_norm)
            y_tensor = torch.from_numpy(y_norm)

            # Define the neural network
            input_size = 1
            hidden_size = 10
            output_size = 1
            model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

            # Define the loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train the network
            num_epochs = 5000
            for epoch in range(num_epochs):
                # Forward pass
                y_pred = model(x_tensor)
                loss = criterion(y_pred, y_tensor)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Save the trained model to disk
            model_path = os.path.join(os.path.dirname(database_path), 'trained_model.pt')
            torch.save(model.state_dict(), model_path)

            # Update the status label
            status_label.config(text="Neural network training completed")
        except:
            # Update the status label if an error occurs
            status_label.config(text="Failed to train neural network")
    
    train_button = tk.Button(root, text='Train Network', command=train_network)
    train_button.pack()
    
    root.mainloop()

if __name__ == '__main__':
    main()
