from nn import *
import csv


class Flat:

    def __init__(self, flat_type: str, city: str, street: str, cost: int, area: int):
        self.flat_type = flat_type
        self.city = city
        self.street = street
        self.cost = cost
        self.area = area

    def __str__(self):
        return "\n".join([
            "Type: " + self.flat_type,
            "City: " + self.city,
            "Street: " + self.street,
            "Cost: %5d Area: %5d" % (self.cost, self.area)
        ])


def train_and_test(data):
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


def read_data(file_name: str):

    with open(file_name, encoding="utf8") as file:

        csv_reader = csv.reader(file, delimiter=",")

        current_line = 0

        column_ordering = {}

        for row in csv_reader:

            if current_line == 0:
                # header

                for i in range(len(row)):
                    column_ordering[row[i]] = i

                current_line += 1

                type_column, city_column, street_column, cost_column, area_column = \
                    column_ordering["type"], column_ordering["city"], column_ordering["street"], \
                    column_ordering["cost"], column_ordering["area"]

            else:

                flat = Flat(
                    row[type_column],
                    row[city_column],
                    row[street_column],
                    int(row[cost_column]),
                    int(row[area_column])
                )

                print(flat)

                current_line += 1

        print("Read %d lines." % current_line)


if __name__ == "__main__":
    read_data("data/berlin.csv")
