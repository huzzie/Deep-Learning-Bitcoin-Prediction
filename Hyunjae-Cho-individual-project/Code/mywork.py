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


# =============================================================================
# RNN Model
# =============================================================================
EPOCHS = 10
BATCH_SIZE = 1
def RNN_model(df, days):
    x_train, y_train, x_test, y_test = split_dataset(days, df)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
    x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    
    # input_shape = (length, features)
    model = Sequential()
    model.add(SimpleRNN(128, input_shape=(x_train.shape[1], 1), return_sequences = True, activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(64, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(32))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'mean_squared_error', optimizer = adam, metrics = ['accuracy'])
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
  model = RNN_model(df, days)
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

def RNN_loss(df, days):
    x_train, y_train, x_test, y_test = split_dataset(days, df)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
    x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    
    # input_shape = (length, features)
    model = Sequential()
    model.add(SimpleRNN(128, input_shape=(x_train.shape[1], 1), return_sequences = True, activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(64, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(32))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'mean_squared_error', optimizer = adam, metrics = ['accuracy'])
    #lstm.summary()
    results = model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)
    plt.plot(results.history['loss'])
    plt.show()

RNN_loss(df2, 3)


# =============================================================================
# RMSE evaluation
# =============================================================================

import math
from sklearn.metrics import mean_squared_error
EPOCHS = 10
BATCH_SIZE = 1
# evaluation
x_train, y_train, x_test, y_test = split_dataset(3, df2)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
model = RNN_model(df2, 3)
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

def evaluation_rmse(days, df):
    x_train, y_train, x_test, y_test = split_dataset(days, df)
    test = x_test[0:, 1]
    res = math.sqrt(mean_squared_error(test, test_predict))
    return res

print(evaluation_rmse(3, df2))


# =============================================================================
# Torch test - RNN
# =============================================================================

def torch_convert(days, df):
    x_train, y_train, x_test, y_test = split_dataset(days, df)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1)) #### Importatnce to reshape!!!!
    x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    return x_test, x_train, y_train, y_test

torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x_test, x_train, y_train, y_test = torch_convert(3, df2)


# hyper-parameters
input_size = 1
hidden_size = 3
output_size = 1
N_EPOCHS= 100
num_layers = 2

# model
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, batch_first = True, nonlinearity = 'tanh')
        self.activation = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden_cell = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, hn = self.rnn(x, (hidden_cell.detach()))
        predict = self.activation(out[:, -1, :]) 
        return predict
rnn_model = RNN(num_layers, hidden_size, output_size, output_size)
# mean squared loss
rnn_loss = torch.nn.MSELoss()
# set optimizer
optimizer = torch.optim.Adam(rnn_model.parameters(), lr = 0.001)

# training
import time
hist = np.zeros(N_EPOCHS)
start= time.time()
rnn = []
for epoch in range(N_EPOCHS):
    y_train_pred = rnn_model(x_train)
    loss = rnn_loss(y_train_pred, y_train)
    print(epoch, 'th epochs', 'value:', loss.item())
    hist[epoch] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print("RNN", time.time() - start)

# evaluation
rnn_model.eval()
predict =  rnn_model(x_test)
plt.plot(y_test)
plt.plot(predict.detach().numpy(), color = 'red')

# =============================================================================
# GRU Torch
# =============================================================================

class GRU(nn.Module):
    def __init__(self, input_size, hidden_layers, num_layers, output_size):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_layers = hidden_layers
        self.gru = nn.GRU(input_size, hidden_layers, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_layers, output_size) # fully connected
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
    def forward(self, x):
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_layers).requires_grad_()
        #internal_state = torch.zeros(self.num_layers, x.size(0), self.hidden_layers).to(device)
        output, hn = self.gru(x, hidden_state.detach())
#         output = self.sigmoid(hn)
        output = self.linear(output[:, -1, :])
        return output
    
x_test, x_train, y_train, y_test = torch_convert(3, df2)
input_size = 1
hidden_layers = 3
num_layers = 2
output_size = 1
N_EPOCHS = 100

model = GRU(input_size = input_size, hidden_layers = hidden_layers, 
            num_layers = num_layers, output_size = output_size)
gru_loss = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training
import time
hist = np.zeros(N_EPOCHS)
start= time.time()
rnn = []
for epoch in range(N_EPOCHS):
    y_train_pred = model(x_train)
    loss = gru_loss(y_train_pred, y_train)
    print(epoch, 'th epochs', 'value:', loss.item())
    hist[epoch] = loss.item()
    optimizer.zero_grad()
    optimizer.step()
    
print("GRU", time.time() - start)

model.eval()
predict =  model(x_test)
plt.plot(y_test)
plt.plot(predict.detach().numpy(), color = 'red')
plt.show()

# loss is not decreasing and training is not working well 

