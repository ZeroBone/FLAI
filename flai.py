from nn import *
import numpy as np
import csv


class Flat:

    type_ids = {}
    city_ids = {}

    def __init__(self, flat_type: str, city: str, street: str, cost: int, area: int):

        self.flat_type = flat_type
        self.city = city
        self.street = street
        self.cost = cost
        self.area = area

        if self.flat_type in Flat.type_ids:
            self.type_id = Flat.type_ids[self.flat_type]
        else:
            self.type_id = len(Flat.type_ids)
            Flat.type_ids[self.flat_type] = self.type_id

        if self.city in Flat.city_ids:
            self.city_id = Flat.city_ids[self.city]
        else:
            self.city_id = len(Flat.city_ids)
            Flat.city_ids[self.city] = self.city_id

    def to_input_vector(self) -> np.ndarray:

        return np.array([
            [float(self.type_id)],
            [float(self.city_id)],
            [float(self.area)]
        ])

    def to_output_vector(self) -> np.ndarray:

        return np.array([
            [float(self.cost)]
        ])

    def __str__(self) -> str:
        return "\n".join([
            "Type: " + self.flat_type,
            "City: " + self.city,
            "Street: " + self.street,
            "Cost: %5d Area: %5d" % (self.cost, self.area)
        ])


def train(training_data: list) -> NeuralNetwork:

    nn = NeuralNetwork(3, [5, 5, 1])

    nn.train(10000, training_data, 0.1, 200)

    return nn


def read_data(file_name: str) -> list:

    data = []

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

                data.append(
                    (
                        flat.to_input_vector(),
                        flat.to_output_vector()
                    )
                )

                current_line += 1

        print("Read %d entries." % current_line)

    return data


def main() -> None:

    data = read_data("data/berlin.csv")

    test_samples = 100

    # everything except the last test_samples elements
    training_data = data[:-test_samples]
    # last test_samples entries
    test_data = data[-test_samples:]

    print("Training data ready, training the neural network...")

    nn = train(training_data)

    print("Training done, testing...")

    mse_avg = 0

    for inp, out in test_data:

        predicted = nn.run(inp)

        mse = (out[0][0] - predicted[0][0]) ** 2

        mse_avg += mse

        print("MSE: %lf" % mse)

    mse_avg /= len(test_data)

    print("Average MSE: %lf" % mse_avg)


if __name__ == "__main__":
    main()
