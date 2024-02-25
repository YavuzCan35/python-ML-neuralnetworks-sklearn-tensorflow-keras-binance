import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib.pyplot as plt
# Load the predictions data from CSV
df = pd.read_csv('predictions.csv')
prices =df['Close'].values.reshape(-1, 1)
# Robust scaling
scaler_robust = RobustScaler()
prices_robust = scaler_robust.fit_transform(prices)

# Mean normalization
scaler_mean = StandardScaler(with_mean=True, with_std=False)
prices_mean = scaler_mean.fit_transform(prices)

# Print the scaled data
print(f'Original prices for :')
print(prices[:100])
print(f'Robust scaled prices for :')
print(prices_robust[:100])
print(f'Mean normalized prices for :')
print(prices_mean[:5])
# Plot the scaled data
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

# Plot original prices
ax1.plot(prices)
ax1.set_title(f'Ori prices fo')

# Plot robust scaled prices
ax2.plot(prices_robust)
ax2.set_title(f'Robust scaled prices for ')

# Plot mean normalized prices
ax3.plot(prices_mean)
ax3.set_title(f'Mean normalized prices for')

# Add axis labels and legend
for ax in (ax1, ax2, ax3):
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Price (USD)')
plt.tight_layout()
plt.show()