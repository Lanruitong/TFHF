import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset, look_back=30):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

path = "/Users/lanruitong/Desktop/天府对冲基金/dataset/stocks.h5"
data = h5py.File(path, 'r')
stock_list = list(data.keys())
number_stocks = len(stock_list)
dataset = data[stock_list[10]]
dataset = np.array(dataset)
dataset = pd.DataFrame(dataset)
dataset['datetime'] = pd.to_datetime(dataset['datetime'],format='%Y%m%d%H%M%S')
data10 = pd.DataFrame()
for i in range(10):
    tmp = data[stock_list[i]]
    tmp = np.array(tmp)
    tmp = pd.DataFrame(tmp)
    tmp['datetime'] = pd.to_datetime(tmp['datetime'],format='%Y%m%d%H%M%S')
    if len(tmp)==3829:
        data10 = pd.concat([data10,tmp])
data10 = data10[['datetime','close']]
data1 = data10[:3829]
data1=data1['close']
scaler = MinMaxScaler(feature_range=(0, 1))
data1 = scaler.fit_transform(np.array(data1).reshape(-1,1))
train_size = int(len(data1)*0.67)
test_size = len(data1)-train_size
train_list = data1[0:train_size]
test_list = data1[train_size:len(data1)]
#test_list = test_list.reset_index().close
X_train,y_train = create_dataset(train_list)
X_test,y_test = create_dataset(test_list)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

model = Sequential()
model.add(LSTM(4,input_shape=(None,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2,shuffle=True)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

# plt.figure(figsize=(10,10))
# plt.plot(y_train,label = '训练集真实')
# plt.plot(train_predict[30:],label = '训练集预测')
#
# plt.show()
plt.plot(y_test,label = 'true')
plt.plot(test_predict,label = 'predict')
plt.legend()
plt.show()