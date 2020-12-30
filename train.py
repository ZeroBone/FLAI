from flai import *
import sys


def train_and_test(dataset: str) -> None:

    data = read_data("data/%s.csv" % dataset)

    data = data_to_vector_pairs(data)

    test_samples = 100

    # everything except the last test_samples elements
    training_data = data[:-test_samples]
    # last test_samples entries
    test_data = data[-test_samples:]

    print("Max cost: %d" % Flat.max_cost)
    print("Max area: %d" % Flat.max_area)
    print("Training data ready, training the neural network...")

    nn = train(training_data)

    print("Training done, saving...")

    nn.save("trained/%s" % dataset)

    print("Neural network saved to 'trained/%s'" % dataset)
    print("Starting tests...")

    run_test(nn, test_data)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not enouph command line arguments specified.")
        print("Please specify the dataset name.")
    else:
        dataset = sys.argv[1]

        print("Dataset: '%s'" % dataset)

        train_and_test(dataset)
