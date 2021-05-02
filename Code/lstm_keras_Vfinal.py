# =============================================================================
# Import libraries
# =============================================================================
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import json
import os
import pandas as pd
from urllib.request import urlopen
import urllib
import requests
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# keras model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers import LSTM, LeakyReLU
from keras.callbacks import CSVLogger, ModelCheckpoint
import tensorflow as tf

import time

# =============================================================================
# Open Poloniex with the API KEy
# =============================================================================
#pip install Poloniex

# from poloniex import Poloniex
# import os
# polo = Poloniex()
# API_KEY = "H1LNX332-FDI51G65-O5QGGEKB-Y0KPIHQV"
# API_SECRET = "0900c0326db91ed0c504eaa12323210005d64f5846e9331a3eac876a2af902ae2f80f65dda626c3936ac9158dad6926fbd58d74931a158a7b6c41e0c07b52388"

# api_key = os.environ.get(API_KEY)
# api_secret = os.environ.get(API_SECRET)
# polo = Poloniex(api_key, api_secret)
# ticker = polo.returnTicker()['BTC_ETH']
# #https://pypi.org/project/poloniex/

# print(ticker)

# #Import Poleniex data
# df = requests.get('https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1405699200&end=9999999999&period=14400')
# df_json = df.json()
# df2 = pd.DataFrame(df_json)
# df2.head()
# df2.to_csv('project_df.csv', encoding = 'utf-8')

# =============================================================================
# Dataset
# =============================================================================
# import the data from the exported dataset
df2 = pd.read_csv('project_df.csv', index_col = 0)

# =============================================================================
# Plot the dataset
# =============================================================================
# Plot the data and define it with a function
def plot_data(dataset):
  plt.figure(figsize=(6,6))
  plt.plot(dataset['open'].values, 'g--',color = 'green', label = 'open')
  plt.plot(dataset['close'].values, 'y--', color = 'yellow', label = 'close')
  plt.plot(dataset['low'].values, color = 'red', label = 'low')
  plt.plot(dataset['high'].values, label = 'high')
  plt.legend()
  plt.xlabel('time')
  plt.ylabel('$ price')
  plt.show()
  
plot_data(df2)

# =============================================================================
# Data preprocessing and split into train and test dataset
# =============================================================================

# Data preprocessing
def data_processing(days, df):
    # min max scaler
    scale = MinMaxScaler()
    columns = ['open', 'close', 'low', 'high']
    for i in columns:
        df[i] = scale.fit_transform(df[i].values.reshape(-1, 1))
    days = days
    # current data
    current = []
    # future data
    future = []
    price = df['close'].values.tolist()
    
    for i in range(len(price) - days):
        current.append([price[i+j] for j in range(days)])
        future.append(price[days+i])
    current, future = np.asarray(current), np.asarray(future)
    return current, future

current, future = data_processing(3, df2)
print(current.shape, future.shape) #(13580, 3) (13580,)

# split the train and test dataset
def split_dataset(days, df):
    current, future = data_processing(days, df)
    train_test_split = int(len(current) * 0.8)
    x_train = current[:train_test_split, :]
    y_train = future[:train_test_split]
    x_test = current[train_test_split:, :]
    y_test = future[train_test_split:]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = split_dataset(3, df2)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) #(10864, 3) (10864,) (2716, 3) (2716,)

# =============================================================================
# LSTM Modeling
# =============================================================================
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

EPOCHS = 10
BATCH_SIZE = 1
def LSTM_model(df, days):
    x_train, y_train, x_test, y_test = split_dataset(days, df)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
    x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    model = Sequential()
    # input_shape = (length, features)
    model.add(LSTM(units=128, input_shape=(x_train.shape[1], 1),return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'tanh'))
    model.compile(loss = 'mse', optimizer = 'adam')
    #lstm.summary()
    model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)
    return model


# =============================================================================
# Evaluation
# =============================================================================
def evaluation(df, days):
  x_train, y_train, x_test, y_test = split_dataset(days, df)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
  x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
  model = LSTM_model(df, days)
  train_predict = model.predict(x_train)
  test_predict = model.predict(x_test)
  return train_predict, test_predict

def evaluation_plot(df, days):
  days = 3
  price = df['close']
  train_test_split = int(len(df) * 0.8)
  train_predict, test_predict = evaluation(df, days)
  plt.plot(price, label = 'actual price') # blue is actual price
  plt.plot(np.arange(days, train_test_split + 1 , 1), train_predict, color = 'red', label = 'train dataset')
  plt.plot(np.arange(train_test_split+ days, train_test_split+ days + len(test_predict), 1), test_predict, 
          color='green', label = 'test dataset')
  plt.legend()
  plt.show()
  
start = time.time()
evaluation_plot(df2, 3)
print(time.time() - start)

# =============================================================================
# Loss plot
# =============================================================================

def LSTM_loss(df, days):
    x_train, y_train, x_test, y_test = split_dataset(days, df)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
    x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    model = Sequential()
    # input_shape = (length, features)
    model.add(LSTM(units=128, input_shape=(x_train.shape[1], 1),return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'tanh'))
    model.compile(loss = 'mse', optimizer = 'adam')
    #lstm.summary()
    results = model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)
    plt.plot(results.history['loss'])
    plt.show()

LSTM_loss(df2, 3)

# =============================================================================
# Evaluation
# =============================================================================
import math
from sklearn.metrics import mean_squared_error
EPOCHS = 10
BATCH_SIZE = 1
# evaluation
x_train, y_train, x_test, y_test = split_dataset(3, df2)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
model = LSTM_model(df2, 3)
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

def evaluation_rmse(days, df):
    x_train, y_train, x_test, y_test = split_dataset(days, df)
    test = x_test[0:, 1]
    res = math.sqrt(mean_squared_error(test, test_predict))
    return res

print(evaluation_rmse(3, df2))

