import numpy as np
import random


def sigma(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigma_derivative(x: float) -> float:
    v = sigma(x)
    return v * (1 - v)


class NeuralNetwork:

    def __init__(self, input_layer_size: int, layer_sizes: list):

        assert input_layer_size >= 1
        assert len(layer_sizes) >= 1

        self.input_layer_size = input_layer_size
        self.layer_sizes = layer_sizes

        self.layer_weights = []

        self.layer_biases = []

        for i in range(self.layer_count()):

            previous_layer_size = self.input_layer_size if i == 0 else layer_sizes[i - 1]
            layer_size = layer_sizes[i]

            assert previous_layer_size >= 1

            # weights[i][j] = weight of the connection between
            # neuron j of the previous layer and
            # neuron i of the current layer
            # the columns of this matrix are therefore the weight vectors
            # and the rows are incoming weights for one neuron in the current layer
            weights = np.random.randn(layer_size, previous_layer_size)

            self.layer_weights.append(weights)

            # biases of neurons in the current layer
            biases = np.random.randn(layer_size, 1)

            self.layer_biases.append(biases)

    def layer_count(self) -> int:
        return len(self.layer_sizes)

    def run(self, mat: np.ndarray) -> np.ndarray:

        (n, m) = mat.shape

        assert n == self.input_layer_size, "Invalid input vector length"

        for i in range(self.layer_count()):

            mat = self.layer_weights[i] @ mat + self.layer_biases[i]

            assert mat.shape[0] == self.layer_sizes[i]
            assert mat.shape[1] == m

            # apply activation function
            mat = sigma(mat)

        return mat

    def run_list(self, inp: list) -> np.ndarray:

        arr = np.array([inp]).T

        return self.run(arr)[:, 0]

    def compute_gradient(self, inp: np.ndarray, out: np.ndarray) -> (np.ndarray, np.ndarray):

        assert inp.shape[0] == self.input_layer_size
        assert inp.shape[1] == 1

        bias_gradient = [np.zeros(bias.shape) for bias in self.layer_biases]
        weight_gradient = [np.zeros(weight.shape) for weight in self.layer_weights]

        # first, we calculate the output of the network for this example
        # and track a and z vectors
        # z values a.k.a. biased weightet sums (without the activation function applied)
        z_vectors = []
        # activations (activation function applied to biased weightet sum)
        a_vectors = [inp]

        for i in range(self.layer_count()):

            inp = self.layer_weights[i] @ inp + self.layer_biases[i]

            z_vectors.append(inp)

            # apply activation function
            inp = sigma(inp)

            a_vectors.append(inp)

        # initialize partial derivatives for the last 2 layers
        # to calculate other derivatives based on them

        dC = (a_vectors[len(a_vectors) - 1] - out) * sigma_derivative(z_vectors[len(z_vectors) - 1])

        bias_gradient[len(bias_gradient) - 1] = dC

        weight_gradient[len(weight_gradient) - 1] = dC @ a_vectors[len(a_vectors) - 2].T

        for layer in range(self.layer_count() - 2, -1, -1):

            z = z_vectors[layer]

            sp = sigma_derivative(z)

            dC = (self.layer_weights[layer + 1].T @ dC) * sp

            bias_gradient[layer] = dC

            weight_gradient[layer] = dC @ a_vectors[layer - 1].T

        return bias_gradient, weight_gradient

    def gradient_descent(self, batch: list, learning_rate: float) -> None:
        # in this method we compute the gradient for every example in the batch
        # with the computed gradient we can determine how the current training
        # example wants to adjust the weights/biases of the neural network
        # however, we cannot just apply the changes directly as in that case
        # the network would consider only aspects relevant for the current training example
        # because of that, we first calculate the average changes all the training
        # examples in this batch "want" to happen, and then we apply them to the
        # neural network

        assert learning_rate > 0

        # we want to average the contribution of every training example
        learning_rate /= len(batch)

        # create a zeroed-out copy of the neural network
        weights_delta = [np.zeros(weight.shape) for weight in self.layer_weights]
        bias_delta = [np.zeros(bias.shape) for bias in self.layer_biases]

        for inp, out in batch:
            # compute how the current training example "wants"
            # to change biases and weights of the neural network

            d_bias, d_weight = self.compute_gradient(inp.copy(), out.copy())

            for i in range(self.layer_count()):
                weights_delta[i] += d_weight[i]
                bias_delta[i] += d_bias[i]

        # we need to subtract the computed gradient from the neural network
        # because intuitively the negative gradient for some entry i represents
        # how sensitive the weight/bias i, or, in other words, how some small change
        # of weight/bias i will affect the cost function that we want to minimize
        for i in range(self.layer_count()):
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
