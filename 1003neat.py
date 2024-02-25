import numpy as np
import pandas as pd
import neat
import talib as ta
import matplotlib.pyplot as plt
import configparser
from tqdm import tqdm

# Load and preprocess data
data = pd.read_csv(r'C:\Users\yavuz\Desktop\Ctest\data180d3m.csv', delimiter=",")
data.columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'qav', 'noot', 'tbbav', 'tbqav', 'ignore']
data = data.iloc[::-1]
data = data[['Close_time', 'Close','High','Low']]
data['Close_time'] = pd.to_datetime(data['Close_time'], unit='ms')
data = data.set_index('Close_time')
data = data.iloc[::-1]

# Define trading simulation function
def simulate_trading(strategy, data, initial_balance=10000, commission=0.001):
    balance = initial_balance
    shares = 0
    trades = []

    for i in range(500):
        if i == 0:
            continue

        # Extract features
        close = data.iloc[i]['Close']
        previous_close = data.iloc[i-1]['Close']
        rsi = ta.RSI(np.array(data['Close']), timeperiod=14)[-1]
        stochastic = ta.STOCH(np.array(data['High']), np.array(data['Low']), np.array(data['Close']))[1][-1]

        # Use strategy to determine action
        action = strategy.activate([close, previous_close, rsi, stochastic])[0]
        position=0
        if action > 0.5 and balance > 0 and position==0:
            # Buy
            shares_to_buy = (balance * (1 - commission)) // close
            cost = shares_to_buy * close
            shares += shares_to_buy
            balance -= cost
            trades.append(('buy', data.index[i], close, shares_to_buy))
            position = 1

        elif action < 0.5 and shares > 0 and position==1:
            # Sell

            print ( action )
            shares_to_sell = shares
            earnings = shares_to_sell * close
            shares = 0
            balance += earnings * (1 - commission)
            trades.append(('sell', data.index[i], close, shares_to_sell))
            position=0

    # Calculate total balance
    total_balance = balance + shares * data.iloc[-1]['Close']
    return trades, total_balance

# Define fitness function
def evaluate_fitness(genomes, config):
    for genome_id, genome in tqdm(genomes, desc="Evaluating fitness"):
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Simulate trading with current strategy
        trades, total_balance = simulate_trading(net, data)

        # Calculate fitness
        genome.fitness = total_balance


        # Print the total balance
        print ( f'Total balance: {total_balance}' )

# Create a ConfigParser object and read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the values in the configuration file
pop_size = config.getint('NEAT', 'pop_size')
fitness_criterion = config.get('NEAT', 'fitness_criterion')
# ...

# Use the values in your code
config_params = neat.config.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward'
)

config_params.pop_size = pop_size
config_params.fitness_criterion = fitness_criterion
# ...

# Create population and run evolution
population = neat.Population(config_params)
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.StdOutReporter(True))
winner = population.run(evaluate_fitness, 50)
# Simulate trading with the winner network
trades, total_balance = simulate_trading(winner, data)

# Plot the closing prices and trades
fig, ax = plt.subplots ( figsize=(16, 9) )
ax.plot ( data.index, data['Close'], label='Closing Prices' )

for trade in trades:
    if trade[0] == 'buy':
        ax.plot ( trade[1], trade[2], 'go', label='Buy' )
    elif trade[0] == 'sell':
        ax.plot ( trade[1], trade[2], 'ro', label='Sell' )

ax.legend ( )
plt.show ( )