import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

# Load the profit data from CSV
df1 = pd.read_csv('profit.csv')

# Plot the profit
plt.plot(df1['Profit'])
plt.show()
# Load the predictions data from CSV
df = pd.read_csv('predictions.csv')

# Plot the predictions
plt.plot(df['Predictions'])
plt.plot(df['Close'])

plt.plot(df1['Profit']*1000+df['Close'])
print(sum(df1['Profit']))
plt.show()