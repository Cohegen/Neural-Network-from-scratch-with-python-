# Neural-Network-from-scratch-with-python-
# Our Neural Network

A simple neural network implemented in Python with:
- 2 inputs
- 1 hidden layer with 2 neurons (h1, h2)
- 1 output neuron (o1)

## Functions

### `sigmoid(x)`
The sigmoid activation function:
$$ f(x) = \frac{1}{1 + e^{-x}} $$

### `deriv_sigmoid(x)`
The derivative of the sigmoid function:
$$ f'(x) = f(x) \times (1 - f(x)) $$
Where \( f(x) \) is the sigmoid function.

### `mse_loss(y_true, y_pred)`
Calculates the mean squared error (MSE) loss between the true labels (`y_true`) and the predicted labels (`y_pred`):
$$ \text{MSE Loss} = \frac{1}{n} \sum (y_{\text{true}} - y_{\text{pred}})^2 $$

## Class: `OurNeuralNetwork`
A simple neural network class with methods to initialize, feedforward, and train the network.

### `__init__(self)`
Initializes the neural network with random weights and biases.

### `feedforward(self, x)`
Performs a forward pass through the network:
1. Calculates the outputs of the hidden neurons (`h1`, `h2`).
2. Calculates the output of the output neuron (`o1`).

### `train(self, data, all_y_trues)`
Trains the neural network using gradient descent:
- `data`: Numpy array of shape (n x 2), where `n` is the number of examples.
- `all_y_trues`: Numpy array of true labels.

## Training the Network

The network is trained on a dataset of 4 examples (`data` and `all_y_trues`):
- `data`: 
- `all_y_trues`: `[1, 0, 0, 1]`

The `train` method performs 1000 epochs of training with a learning rate of 0.01. The loss is printed every 10 epochs.

## Making Predictions

After training, the network is used to make predictions for two new examples:
- `emily`: `[-7, -3]`
- `frank`: `[20, 2]`

The predictions are printed with 3 decimal places.

## Example Usage
```python
import numpy as np

# Define sigmoid, deriv_sigmoid, mse_loss functions
# Define OurNeuralNetwork class

# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1,  # Alice
  0,  # Bob
  0,  # Charlie
  1,  # Diana
])

# Train the neural network
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# Make predictions
emily = np.array([-7, -3])
frank = np.array([20, 2])
print(f"Emily: {network.feedforward(emily):.3f}")
print(f"Frank: {network.feedforward(frank)::.3f}")
