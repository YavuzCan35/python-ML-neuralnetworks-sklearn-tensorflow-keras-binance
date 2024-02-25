import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import talib as ta
# Load the data
df = pd.read_csv(r'C:\Users\yavuz\Desktop\Ctest\data180d3m.csv', delimiter=",")
df.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'qav', 'noot','tbbav','tbqav', 'ignore']
# Drop unnecessary columns
df = df.drop(['Open_time', 'Close_time', 'qav', 'noot', 'tbbav', 'tbqav', 'ignore'], axis=1)

# Define the binary output based on the price movement
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Drop unnecessary columns
df = df.drop(['Open', 'High', 'Low', 'Volume'], axis=1)

# Preprocess and engineer features
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df = df.dropna()

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Define the number of time steps to use in each training example
n_steps = 64
n_features = 3

# Create sequences of data for input into LSTM
def create_sequences(data, n_steps):
    X = []
    y = []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, :])
        y.append(data[i, -1])
    X, y = np.array(X), np.array(y)
    return X, y

X, y = create_sequences(scaled_data, n_steps)

# Split data into training, validation, and test sets
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

train_size = int(len(X)*train_ratio)
val_size = int(len(X)*val_ratio)
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(32, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

# Fit the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])

# Make predictions on the test set
y_pred = model.predict_classes(X_test)

# Print classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plot the test set predictions and actual values
import matplotlib.pyplot as plt
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
