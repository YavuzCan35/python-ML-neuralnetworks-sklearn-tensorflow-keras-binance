import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

import pandas as pd
# Load the dataset
df = pd.read_csv(r'C:\Users\yavuz\Desktop\Ctest\data180d3m.csv', delimiter=",")
# Rename the columns
df = pd.DataFrame(df)
"""print(df.iloc[0,:])
exit()"""
df.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'qav', 'noot','tbbav','tbqav', 'ignore']
# Load the additional input data
#df2 = pd.read_csv(r'path/to/additional_input_data.csv', delimiter=",")

# Preprocess the data
X = df.iloc[:, [ 1, 2, 3, 4, 5, 6]].values # Select the first 7 columns as input data
#X2 = df2.iloc[:, [0]].values # Select the first column of the additional input data
#X = np.concatenate((X, X2), axis=1) # Concatenate the inputs along the column axis
y = df.iloc[:, 4].values # The target data is the close value
# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Reshape the input data to match the input shape expected by the LSTM model
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if os.path.exists('model.h5'):
    # Load the saved model
    ask=input("Would you like to reset the model ? y for yes, n for no!:\n")
    if ask=="y":
        # Define the model architecture
        inputs = tf.keras.layers.Input ( shape=(1, 6,))
        x = tf.keras.layers.LSTM ( 1280, activation='relu', return_sequences=True ) ( inputs )
        x = tf.keras.layers.LSTM ( 1280, activation='relu' ) ( x )
        outputs = tf.keras.layers.Dense ( 2, activation='linear' ) ( x )
        model = tf.keras.Model ( inputs=inputs, outputs=outputs )
        model.compile ( optimizer='adam', loss='mean_squared_error', metrics=['accuracy'] )
    else:
        model = tf.keras.models.load_model ( 'model.h5' )
        pass
else:
    # Define the model architecture
    inputs = tf.keras.layers.Input ( shape=(1, 6,) )
    x = tf.keras.layers.LSTM ( 128, activation='relu', return_sequences=True ) ( inputs )
    x = tf.keras.layers.LSTM ( 128, activation='relu' ) ( x )
    outputs = tf.keras.layers.Dense ( 1, activation='linear' ) ( x )
    model = tf.keras.Model ( inputs=inputs, outputs=outputs )
    model.compile ( optimizer='adam', loss='mean_squared_error', metrics=['accuracy'] )


import matplotlib.pyplot as plt
# Start online training
num_epochs = 10  # number of times to loop over all data
batch_size = 32  # number of samples per gradient update
def get_new_data():
    # Read the new data from the CSV file
    df =pd.read_csv(r'C:\Users\yavuz\Desktop\Ctest\data180d3m.csv', delimiter=",")

    # Preprocess the new data
    X = df.iloc[:, [1, 2, 3, 4, 5, 6]].values # Select the first 7 columns as input data
    y = df.iloc[:, 4].values # The target data is the close value
    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # Reshape the input data to match the input shape expected by the LSTM model
    X = np.reshape ( X, (X.shape[0], 1, X.shape[1]) )
    return X, y



plt.figure ( 0 )
#plt.ion ( )
for epoch in range ( num_epochs ):
    preds = []
    actuals = []

    # Get new data
    X_new, y_new = get_new_data ( )
    predictions = model.predict ( X_new )
    for i in range ( len ( predictions ) ):
        print ( 'Predicted:', predictions[i][0], 'Actual:', y_new[i] )
        print ( 'Predicted:', predictions[i][1], 'Actual:', y_new[i] )
        preds.append ( predictions[i][0]*1000000+ y_new[i] )
        actuals.append ( y_new[i] )
    #history = model.fit ( X_train, y_train, batch_size=batch_size, epochs=1 )
    # Plot the training loss
    plt.clf ( )
    plt.plot ( preds, label='pred' )
    plt.plot ( actuals, label='real' )

    plt.legend ( )
    plt.pause (1)

    plt.show ( )
    input("Press any key to repeat the training for the given number of epoch times")
    # Train the model
    model.save ( 'model.h5' )
"""    latest_row_of_X = X_new[-1, :]
    prediction_for_next_row = model.predict ( latest_row_of_X.reshape ( 1, -1 ) )
"""


exit()
# Train the model
import matplotlib.pyplot as plt
for epoch in range(10):
    print(f"Epoch:{epoch+1}")
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
    print(X_test)
    y_pred = model.predict ( np.reshape ( X_input, (X_input.shape[0], X_input.shape[1], 1) ) )
    diff = y_test - y_pred[:,0]
    #plt.plot(diff, 'g', label='Difference between Predicted and Real Data')
    plt.plot ( y_test, 'r', label='Real Target Data' )
    plt.plot ( y_pred[:, 0], 'b', label='Predicted Target Data' )
    plt.plot ( y_pred[:, 1], 'g', label='Predicted Target Data' )
    plt.pause(0.05)
    plt.legend()
    plt.show()
"""
    plt.legend()
    plt.show()"""
# Save the model
#model.save ( 'model.h5' )
# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: {}'.format(test_acc))
