import numpy as np
import random


def sigma(x: float) -> float:
    return 1 / (1 + np.exp(x))


def sigma_derivative(x: float) -> float:
    return sigma(x) * (1 - sigma(x))


class NeuralNetwork:

    def __init__(self, input_layer_size: int, layer_sizes: list):

        assert input_layer_size >= 1
        assert len(layer_sizes) >= 1

        self.input_layer_size = input_layer_size
        self.layer_sizes = layer_sizes

        self.layer_weights = []

        self.layer_biases = []

        for i in range(self.layer_count()):

            layer_size = self.input_layer_size if i == 0 else layer_sizes[i - 1]
            next_layer_size = layer_sizes[i]

            assert layer_size >= 1

            # weights[i][j] = weight of the connection between
            # neuron i of the current layer and
            # neuron j of the next layer
            # the columns of this matrix are therefore the weight vectors
            weights = np.random.randn(next_layer_size, layer_size)

            self.layer_weights.append(weights)

            # biases of neurons in the current layer
            biases = np.random.randn(layer_size)

            self.layer_biases.append(biases)

    def layer_count(self) -> int:
        return len(self.layer_sizes)

    def run(self, mat):

        (n, m) = mat.shape

        assert n == self.input_layer_size, "Invalid input vector length"

        for i in range(self.layer_count()):

            mat = self.layer_weights[i] @ mat

            assert mat.shape[0] == self.layer_sizes[i]

            # the linear combination with neuron connection weights has been computed
            # so we only need to add bias terms and compute sigma of the result
            for neuron in range(self.layer_sizes[i]):
                mat[neuron] = sigma(mat[neuron] + self.layer_biases[i][neuron])

        return mat

    def run_single(self, inp):
        return self.run(inp.transpose())

    def compute_gradient(self, inp, out):
        # TODO
        return [], []

    def gradient_descent(self, batch: list, learning_rate: float):
        # in this method we compute the gradient for every example in the batch
        # with the computed gradient we can determine how the current training
        # example wants to adjust the weights/biases of the neural network
        # however, we cannot just apply the changes directly as in that case
        # the network would consider only aspects relevant for the current training example
        # because of that, we first calculate the average changes all the training
        # examples in this batch "want" to happen, and then we apply them to the
        # neural network

        assert learning_rate > 0

        # create a zeroed-out copy of the neural network
        weights_delta = [np.zeros(weight.shape) for weight in self.layer_weights]
        bias_delta = [np.zeros(bias.shape) for bias in self.layer_biases]

        for inp, out in batch:
            # delta bias and delta weight
            d_bias, d_weight = self.compute_gradient(inp, out)

            for i in range(self.layer_count()):
                weights_delta[i] += d_weight[i]
                bias_delta[i] += d_bias[i]

        # we need to subtract the computed gradient from the neural network
        # because intuitively the negative gradient for some entry i represents
        # how sensitive the weight/bias i, or, in other words, how some small change
        # of weight/bias i will affect the cost function that we want to minimize
        for i in range(self.layer_count()):
            # compute the average delta
            weights_delta[i] /= len(batch)
            bias_delta[i] /= len(batch)
            # apply the average delta to the neural network
            self.layer_weights[i] -= learning_rate * weights_delta[i]
            self.layer_biases[i] -= learning_rate * bias_delta[i]

    def train(self, epochs: int, data: list, learning_rate: float, batch_size: int) -> None:

        assert learning_rate > 0

        length = len(data)

        for e in range(epochs):

            # shuffle the data to then partition it into batches
            random.shuffle(data)

            # we are approximating the overall average gradient
            # with the average gradient of a randomnly-chosen batch,
            # partition data in batches for more efficient gradient calculation
            for start in range(0, length, batch_size):
                self.gradient_descent(data[start:start + batch_size], learning_rate)

            print("Training epoch %d ended!" % (e + 1))
