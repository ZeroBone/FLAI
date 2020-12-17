from nn import *


def train():
    nn = NeuralNetwork(2, [1])

    training_data = [
        (
            np.array([[0.0], [0.0]]),
            np.array([[1.0]])
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([[1.0]])
        ),
        (
            np.array([[1.0], [0.0]]),
            np.array([[0.0]])
        ),
        (
            np.array([[1.0], [1.0]]),
            np.array([[1.0]])
        )
    ]

    nn.train(100, training_data, 50, 4)

    print(nn.run_list([0.0, 0.0]))
    print(nn.run_list([0.0, 1.0]))
    print(nn.run_list([1.0, 0.0]))
    print(nn.run_list([1.0, 1.0]))


if __name__ == "__main__":
    train()
