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

def load_csv_file(file_path):
    # Проверяем существует ли файл базы данных с таким же именем, как и csv-файл
    db_path = os.path.splitext(file_path)[0] + '.db'
    if os.path.exists(db_path):
        # Если файл существует, подключаемся к базе данных
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        print("Подключение к базе данных установлено")
    else:
        # Если файла нет, создаем новую базу данных
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        # создаем таблицу с полями: дата, цена открытия, цена закрытия, высшая цена, низшая цена, объем
        c.execute('''CREATE TABLE stocks (date text, open real, close real, high real, low real, volume real)''')
        print("Новая база данных создана")

    # Загружаем csv-файл в базу данных
    df = pd.read_csv(file_path, delimiter=',', index_col=0, parse_dates=True)
    df.to_sql('stocks', conn, if_exists='append')
    print("Файл успешно загружен в базу данных")
    conn.commit()
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
        balance = float(balance_field.get())
        percent = float(percent_field.get())
        print('Balance:', balance)
        print('Percent:', percent)
        file_path = filedialog.askopenfilename()
        print('Selected file:', file_path)
        db_file = filedialog.asksaveasfilename(defaultextension='.db')
        print('Selected database:', db_file)
        train_neural_network(file_path, db_file)
    
    train_button = tk.Button(root, text='Train Network', command=train_network)
    train_button.pack()
    
    root.mainloop()

if __name__ == '__main__':
    main()

