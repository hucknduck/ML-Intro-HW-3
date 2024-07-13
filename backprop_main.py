import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

# Loading Data
np.random.seed(0)  # For reproducibility
n_train = 50000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

learning_rates = [0.001, 0.01, 0.1, 1, 10]

Params = []
train_costs = []
test_costs = []
train_accs = []
test_accs = []


# Training configuration
epochs = 30
batch_size = 100
layer_dims = [784, 40, 10]

for l in range(len(learning_rates)):
    learning_rate = learning_rates[l]

    net = Network(layer_dims)
    param, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

    Params.append(param)
    train_costs.append(epoch_train_cost)
    test_costs.append(epoch_test_cost)
    train_accs.append(epoch_train_acc)
    test_accs.append(epoch_test_acc)

Colors = ["orange", "blue", "red", "purple", "green"]
EpochsScale = np.array(range(30))
for l in range(len(learning_rates)):
    plt.figure(1)
    plt.plot(EpochsScale, train_accs[l], color=Colors[l])
    plt.grid(True)
    
    plt.figure(2)
    plt.plot(EpochsScale, train_costs[l], color=Colors[l])
    plt.grid(True)
    
    plt.figure(3)
    plt.plot(EpochsScale, test_accs[l], color=Colors[l])
    plt.grid(True)
    
plt.figure(1)
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.show

plt.figure(2)
plt.xlabel('Training Cost')
plt.show

plt.figure(3)
plt.xlabel('Test Accuracy')
plt.show

