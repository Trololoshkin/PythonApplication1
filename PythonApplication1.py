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

 
def train_neural_network(file_path, db_file):
    data = load_csv_file(file_path)
    create_database(db_file)
    # TODO: train neural network

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
    
    def train_network():
    # check if csv file and database are selected
    if not csv_file_path:
        status_label.config(text="Please select a CSV file")
        return
    if not db_conn:
        status_label.config(text="Please connect to a database")
        return
    
    # read csv file
    try:
        data = pd.read_csv(csv_file_path)
    except:
        status_label.config(text="Failed to read CSV file")
        return
    
    # preprocess data
    x, y = preprocess_data(data)
    
    # split data into train and test sets
    x_train, x_test, y_train, y_test = split_data(x, y)
    
    # create neural network model
    input_dim = len(x.columns)
    output_dim = 2
    hidden_dim = 100
    model = create_model(input_dim, hidden_dim, output_dim)
    
    # train neural network model
    learning_rate = 0.001
    epochs = 1000
    train_model(model, x_train, y_train, learning_rate, epochs)
    
    # test neural network model
    test_model(model, x_test, y_test)
    
    # save neural network model to database
    model_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    model_table_name = "models"
    save_model_to_db(model, model_name, model_table_name, db_conn)
    
    # update status label
    status_label.config(text="Training complete")
    
    train_button = tk.Button(root, text='Train Network', command=train_network)
    train_button.pack()
    
    root.mainloop()

if __name__ == '__main__':
    main()
