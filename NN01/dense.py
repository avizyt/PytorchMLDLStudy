import numpy as np


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # initialize the weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_Relu:
    def forward(self, inputs):
        self.ouput = np.maximum(0, inputs)


class Activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
