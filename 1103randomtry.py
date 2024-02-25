import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

# Read in data and rename columns
data = pd.read_csv(r'C:\Users\yavuz\Desktop\Ctest\data180d3m.csv', delimiter=",")
data.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'qav', 'noot', 'tbbav', 'tbqav', 'ignore']

# Define hyperparameters
WINDOW_SIZE = 10
PROFIT_RATIO = 2
STOP_LOSS_RATIO = 1.5

# Compute Sharpe Ratio
returns = np.log(data['Close']).diff()
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())

# Compute technical indicators
data['RSI'] = talib.RSI(data['Close'])
data['MACD'], data['MACD_SIGNAL'], _ = talib.MACD(data['Close'])
data['BB_UPPER'], data['BB_MIDDLE'], data['BB_LOWER'] = talib.BBANDS(data['Close'])

# Define function to compute trade signals
def get_trade_signal(data):
    buy_signal = False
    sell_signal = False
    stop_loss_price = 0
    profit_price = 0

    if data['Close'] > data['BB_UPPER'] and data['RSI'] > 70 and data['MACD'] > data['MACD_SIGNAL']:
        buy_signal = True
        stop_loss_price = data['Close'] / STOP_LOSS_RATIO
        profit_price = data['Close'] * PROFIT_RATIO
    elif data['Close'] < data['BB_LOWER'] and data['RSI'] < 40 and data['MACD'] < data['MACD_SIGNAL']:
        sell_signal = True
        stop_loss_price = data['Close'] * STOP_LOSS_RATIO
        profit_price = data['Close'] / PROFIT_RATIO

    return buy_signal, sell_signal, stop_loss_price, profit_price

# Define function to execute trades
def execute_trade(data, trade_type):
    global balance
    global num_trades
    global profit

    if trade_type == 'buy':
        num_shares = balance / data['Close']
        balance = 0
        stop_loss_price = data['stop_loss_price']
        profit_price = data['profit_price']
    elif trade_type == 'sell':
        num_shares = -1 * balance
        balance = balance + (num_shares * data['Close'])
        stop_loss_price = data['stop_loss_price']
        profit_price = data['profit_price']

    num_trades += 1

    while True:
        if data['Low'] <= stop_loss_price:
            balance += num_shares * stop_loss_price
            profit -= num_shares * (data['Close'] - stop_loss_price)
            break
        elif data['High'] >= profit_price:
            balance += num_shares * profit_price
            profit += num_shares * (profit_price - data['Close'])
            break
        else:
            pass

# Initialize variables
balance = 10000
num_trades = 0
profit = 0
trades = []
trade_history = []

# Iterate over each row of data
for i in range(len(data)):
    # Add row to window
    window = data.iloc[max(0, i - WINDOW_SIZE + 1):i+1]

    # Get trade signal
    trade_signal = get_trade_signal ( window.iloc[-1] )

    # Execute trade
    if trade_signal[0]:
        execute_trade ( window.iloc[-1], 'buy' )
        trades.append ( 'buy' )
    elif trade_signal[1]:
        execute_trade ( window.iloc[-1], 'sell' )
        trades.append ( 'sell' )
    else:
        trades.append ( 'hold' )

    # Save trade history
    trade_history.append ( balance )
print('Final balance:', balance)
print('Number of trades:', num_trades)
print('Total profit:', profit)

plt.plot(trade_history)
plt.title('Trade History')
plt.xlabel('Trade')
plt.ylabel('Balance')
plt.show()
plt.plot(data['Close'])
plt.plot(data['BB_UPPER'])
plt.plot(data['BB_MIDDLE'])
plt.plot(data['BB_LOWER'])
plt.title('Bollinger Bands')
plt.legend(['Close', 'Upper Band', 'Middle Band', 'Lower Band'])
plt.show()

plt.plot(data['RSI'])
plt.title('RSI')
plt.xlabel('Time')
plt.ylabel('RSI')
plt.show()

plt.plot(data['MACD'])
plt.plot(data['MACD_SIGNAL'])
plt.title('MACD')
plt.legend(['MACD', 'Signal Line'])
plt.show()
plt.plot(data['Close'])
for i in range(len(trades)):
    if trades[i] == 'buy':
        plt.scatter(i, data['Close'][i], color='green')
    elif trades[i] == 'sell':
        plt.scatter(i, data['Close'][i], color='red')
plt.title('Trades')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()