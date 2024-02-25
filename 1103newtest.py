import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,RobustScaler
import sklearn

# Read in data and rename columns
data = pd.read_csv(r'C:\Users\yavuz\Desktop\Ctest\data180d3m.csv', delimiter=",")
data.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'qav', 'noot', 'tbbav', 'tbqav', 'ignore']

# Drop unnecessary columns
data = data.drop(['Open_time', 'Close_time', 'qav', 'noot', 'tbbav', 'tbqav', 'ignore'], axis=1)
print(data)
# Convert to numpy array
data = data.values

# Create a function to generate the features
def generate_features(data):
    data_cp = np.copy(data)
    data_cp = pd.DataFrame(data_cp, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    data_cp['Open-Close'] = data_cp.iloc[:,1] - data_cp.iloc[:,4]
    data_cp['High-Low'] = data_cp.iloc[:,2] - data_cp.iloc[:,3]
    return data_cp

# Generate the features
data_features = generate_features(data)

# Create a function to generate the labels
def generate_labels(data):
    # Create a copy of the data
    data_cp = data.copy()
    # Convert the data to a Pandas DataFrame
    data_cp = pd.DataFrame(data_cp, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    # Create the labels
    data_cp['Above_Close'] = np.where(data_cp['Close'].shift(-1)*1.01 > data_cp['High'], 1, 0)
    # Return the result
    return data_cp.values

# Generate the labels
data_labels = generate_labels(data)

# Create a function to split the data
def split_data(data, training_size=0.8):
    # Split the data
    train = data[:int(training_size*len(data))]
    test = data[int(training_size*len(data)):]
    # Return the result
    return train, test

# Split the data
train_features, test_features = split_data(data_features)
train_labels, test_labels = split_data(data_labels)

# Create a function to scale the data
def scale_data(train, test):
    # Fit the scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    # Return the result
    return scaler, train_scaled, test_scaled


# Scale the data
scaler, train_scaled, test_scaled = scale_data(train_features, test_features)

# Create a function to create the data sets
def create_data_sets(train, labels, time_steps):
    # Create the data sets
    X, y = [], []
    for i in range(len(train)-time_steps-1):
        data = train[i:(i+time_steps), :]
        if data.shape != (time_steps, train.shape[1]):
            continue
        X.append(data)
        y.append(labels[i+time_steps, -1])
    # Convert to numpy arrays
    X, y = np.array(X), np.array(y)
    # Return the result
    return X, y



# Create the data sets
time_steps=60
X_train, y_train = create_data_sets(train_scaled, train_labels,time_steps)
num_features = X_train.shape[2]
print(num_features)
X_test, y_test = create_data_sets(test_scaled, test_labels,time_steps)

# Create a function to create the model
def create_model(time_steps, num_features, neurons=32):
    # Create the model
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(time_steps, num_features)))
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Return the model
    return model

# Create the model
model = create_model(time_steps, num_features, neurons=32)

# Create a function to fit the model
def fit_model(model, X_train, y_train, epochs, batch_size):
    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    # Return the model
    return model

# Fit the model
model = fit_model(model, X_train, y_train, epochs=1, batch_size=32)

# Create a function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    # Return the accuracy
    return accuracy
# Save the model
model.save('model5.h5')
# Evaluate the model
accuracy = evaluate_model(model, X_test, y_test)
print('Accuracy:',accuracy)
# Print the accuracy
print('Accuracy: %.2f' % (accuracy*100))

# Create a function to make predictions
def make_predictions(model, data, time_steps):
    # Create a copy of the data
    data_cp = data.copy()
    # Create a new empty list
    predictions = []
    # Loop through each row of data
    for i in range(len(data_cp)):
        # Reshape the data
        curr_frame = data_cp[i, 0].reshape((1, time_steps, 1))
        # Get the prediction
        predicted = model.predict(curr_frame)[0,0]
        # Append the result
        predictions.append(predicted)
    # Return the result
    return predictions

# Make predictions
predictions = make_predictions(model, X_test, X_test.shape[1])
# Save predictions to CSV
df_predictions = pd.DataFrame(predictions, columns=['Predictions'])
df_test_labels = pd.DataFrame(test_labels, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Above_Close'])
df_combined = pd.concat([df_test_labels, df_predictions], axis=1)
df_combined.to_csv('predictions.csv', index=False)
# Load the predictions data from CSV
df = pd.read_csv('predictions.csv')

# Plot the predictions
plt.plot(df['Predictions'])
plt.plot(df['Close'])
plt.show()
# Create a function to calculate the profit
def calculate_profit(predictions, test_labels):
    # Create a copy of the labels
    test_labels_cp = np.copy(test_labels)
    test_labels_cp = pd.DataFrame(test_labels_cp, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Above_Close'])
    test_labels_cp['Above_Close'] = test_labels_cp['Above_Close'].astype(int)
    # Create a new empty list
    profits = []
    # Loop through the predictions
    for i in range(len(predictions)):
        # Calculate the profit
        curr_profit = ((test_labels_cp.iloc[i+1]['Close'] - test_labels_cp.iloc[i]['Close']) / test_labels_cp.iloc[i]['Close']) * 100
        # Append the profit to the list
        profits.append(curr_profit)
    # Return the result
    return profits

# Calculate the profit
profit = calculate_profit(predictions, test_labels)
# Save profit to CSV
df_profit = pd.DataFrame(profit, columns=['Profit'])
df_profit.to_csv('profit.csv', index=False)
# Print the profit
print('Profit: $%.2f' % sum(profit))
# Load the profit data from CSV
df = pd.read_csv('profit.csv')

# Plot the profit
plt.plot(df['Profit'])
plt.show()

# Create a function to plot the profit
def plot_profit(profit):
    # Create a copy of the profit
    profit_cp = profit.copy()
    # Insert a 0 at the beginning of the list
    profit_cp.insert(0, 0)
    # Loop through each value and cumulatively sum it
    for i in range(1, len(profit_cp)):
        profit_cp[i] = profit_cp[i] + profit_cp[i-1]
    # Plot the profit
    plt.plot(profit_cp)
    plt.show()

# Plot the profit
plot_profit(profit)

# Create a function to plot the predictions
def plot_predictions(predictions, test_labels):
    # Convert test_labels to a pandas dataframe
    test_labels_cp = pd.DataFrame(test_labels, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Above_Close'])
    # Create a new empty list
    close = []
    # Loop through each prediction and label
    for i in range(len(predictions)):
        # Append the close price
        close.append(test_labels_cp.iloc[i]['Close'])
    # Plot the predictions
    plt.plot(predictions)
    plt.plot(close)
    plt.show()

# Plot the predictions
plot_predictions(predictions, test_labels)
