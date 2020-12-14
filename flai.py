from nn import *


def train():
    nn = NeuralNetwork(4, [2, 1])

    print(nn.run(np.zeros((4, 10))))


if __name__ == "__main__":
    train()
