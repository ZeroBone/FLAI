from ai.imports import *


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

    def layer_count(self):
        return len(self.layer_sizes)

    def run(self, mat):

        (n, m) = mat.shape

        assert n == self.input_layer_size, "Invalid input vector length"

        for i in range(self.layer_count()):

            mat = self.layer_weights[i] @ mat

            assert mat.shape[0] == self.layer_sizes[i]

            for neuron in range(self.layer_sizes[i]):
                mat[neuron] = sigma(mat[neuron] + self.layer_biases[i][neuron])

        return mat
