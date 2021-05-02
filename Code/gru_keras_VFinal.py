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
from keras.layers import SimpleRNN
from keras import optimizers
from keras.utils import np_utils
from keras.layers import LSTM, LeakyReLU
from keras.callbacks import CSVLogger, ModelCheckpoint
import tensorflow as tf
from keras import backend as K
import time

# =============================================================================
# Dataset
# =============================================================================
# import the data from the exported dataset
df2 = pd.read_csv('project_df.csv', index_col = 0)


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
# GRU
# =============================================================================

EPOCHS = 10
BATCH_SIZE = 1
from keras.layers import GRU, Dense, Dropout
def model_GRU(df, days):
    x_train, y_train, x_test, y_test = split_dataset(days, df)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
    x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    
    model = Sequential()
    model.add(GRU(units = 128, input_shape=(x_train.shape[1], 1), return_sequences = True, activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(64, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'mse', optimizer = adam, metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)
    #lstm.summary()
    return model
# =============================================================================
# Evaluation
# =============================================================================
def evaluation(df, days):
  x_train, y_train, x_test, y_test = split_dataset(days, df)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
  x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
  model = model_GRU(df, days)
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
print('GRU takes:', time.time() - start)



# =============================================================================
# Evalulate RMSE
# =============================================================================
import math
from sklearn.metrics import mean_squared_error

start = time.time()
x_train, y_train, x_test, y_test = split_dataset(3, df2)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
model = model_GRU(df2, 3)
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)
print('GRU takes:', time.time() - start)

def evaluation_rmse(days, df):
    x_train, y_train, x_test, y_test = split_dataset(days, df)
    test = x_test[0:, 1]
    res = math.sqrt(mean_squared_error(test, test_predict))
    return res

print(evaluation_rmse(3, df2))

scale = MinMaxScaler()
def r2_calculate(actual, predicted):
    num = K.sum(K.square(actual - K.mean(actual)))
    res = K.sum(K.square(actual - predicted))
    return (1- res/(num + K.epsilon()))


# =============================================================================
# Plot loss
# =============================================================================

def GRU_loss(df, days):
    x_train, y_train, x_test, y_test = split_dataset(days, df)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
    x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    
    model = Sequential()
    model.add(GRU(units = 128, input_shape=(x_train.shape[1], 1), return_sequences = True, activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(64, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'mse', optimizer = adam, metrics = ['accuracy'])
    results = model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)
    plt.plot(results.history['loss'])
    plt.show()

GRU_loss(df2, 3)





