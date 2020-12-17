from flai import *
import sys


def test(nn_name: str, dataset: str):

    nn = NeuralNetwork.load("trained/%s.npz" % nn_name)

    data = read_data("data/%s.csv" % dataset)

    print("Starting tests...")

    run_test(nn, data)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not enouph command line arguments specified.")
        print("Please specify the dataset name.")
    else:
        dataset = sys.argv[1]

        print("Dataset: '%s'" % dataset)

        test(dataset, dataset)
