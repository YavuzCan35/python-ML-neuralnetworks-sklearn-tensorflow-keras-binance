import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM ,Input,Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,RobustScaler
import sklearn

data = pd.read_csv(r'C:\Users\yavuz\Desktop\Ctest\data180d3m.csv', delimiter=",")
data.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'qav', 'noot', 'tbbav', 'tbqav', 'ignore']
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

X = data.values
scaler = RobustScaler()
X = scaler.fit_transform(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

def calculate_returns(X):
    returns = np.zeros(X.shape[0])
    for i in range(1, X.shape[0]):
        # long
        if X[i-1, 3] < X[i, 3]:
            returns[i] = X[i, 3] /X[i-1, 3] - 0.002 -1
        # short
        else:
            returns[i] = X[i-1, 3] / X[i, 3] - 0.002 -1
    return returns
"""
X.shape[0] : the number of samples
X.shape[1] : the number of features (Open, High, Low, Close, Volume)
X.shape[2] : the number of predictors (1 in this case)
X[i, 0] : Open value of the i-th sample
X[i, 1] : High value of the i-th sample
X[i, 2] : Low value of the i-th sample
X[i, 3] : Close value of the i-th sample
X[i, 4] : Volume value of the i-th sample
"""

epochs = 10
plt.figure(figsize=(15,6))
predictions = np.zeros(X.shape[0])
for epoch in range(1, epochs+1):
    model.fit(X, calculate_returns(X), epochs=1, batch_size=32, verbose=2)
    predictions = model.predict(X)

plt.plot(predictions, label='Epoch '+str(1))


plt.scatter(range(X.shape[0]), data['Close'].values, c='r')
plt.legend()
plt.show()
