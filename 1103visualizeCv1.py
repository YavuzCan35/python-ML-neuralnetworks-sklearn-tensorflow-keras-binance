import matplotlib.pyplot as plt

# read the output file
with open(r'C:\Users\yavuz\Desktop\Ctest\predicted_values.txt', 'r') as f:
    y = [float(line.strip()) for line in f.readlines()]
# read the output file
with open(r'C:\Users\yavuz\Desktop\Ctest\data180d3m.csv', 'r') as f:
    data = f.readlines()
    x = [float(line.split(',')[0]) for line in data]
    y1 = [float(line.split(',')[1]) for line in data]
# plot the output
plt.plot(y1,"blue")

# plot the output
plt.plot(y,"black")
plt.xlabel('Time Step')
plt.ylabel('Output Value')
plt.show()
