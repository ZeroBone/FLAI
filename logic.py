from nn import *


def and_gate():
    nn = NeuralNetwork(2, [1])

    training_data = [
        (
            np.array([[0.0], [0.0]]),
            np.array([[0.0]])
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([[0.0]])
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


def or_gate():
    nn = NeuralNetwork(2, [1])

    training_data = [
        (
            np.array([[0.0], [0.0]]),
            np.array([[0.0]])
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([[1.0]])
        ),
        (
            np.array([[1.0], [0.0]]),
            np.array([[1.0]])
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


def xor_gate():
    nn = NeuralNetwork(2, [4, 2, 1])

    training_data = [
        (
            np.array([[0.0], [0.0]]),
            np.array([[0.0]])
        ),
        (
            np.array([[0.0], [1.0]]),
            np.array([[1.0]])
        ),
        (
            np.array([[1.0], [0.0]]),
            np.array([[1.0]])
        ),
        (
            np.array([[1.0], [1.0]]),
            np.array([[0.0]])
        )
    ]

    nn.train(1000, training_data, 0.01, 4)

    print(nn.run_list([0.0, 0.0]))
    print(nn.run_list([0.0, 1.0]))
    print(nn.run_list([1.0, 0.0]))
    print(nn.run_list([1.0, 1.0]))


if __name__ == "__main__":
    print("AND gate:")
    and_gate()
    print("OR gate:")
    or_gate()
    print("XOR gate:")
    xor_gate()
