from nn import *
import numpy as np
import csv


class Flat:

    type_ids = {}
    city_ids = {}

    max_area = 0
    max_cost = 0

    def __init__(self, flat_type: str, city: str, street: str, cost: int, area: int):

        assert area > 0, "Invalid area"
        assert cost > 0, "Invalid cost"

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

        if self.area > Flat.max_area:
            Flat.max_area = self.area

        if self.cost > Flat.max_cost:
            Flat.max_cost = self.cost

    def to_input_vector(self) -> np.ndarray:

        assert Flat.max_area > 0

        return np.array([
            [self.type_id / len(Flat.type_ids)],
            [self.city_id / len(Flat.city_ids)],
            [self.area / Flat.max_area]
        ])

    def to_output_vector(self) -> np.ndarray:

        return np.array([
            [Flat.encode_cost(self.cost)]
        ])

    @staticmethod
    def encode_cost(cost: int) -> float:
        assert Flat.max_cost > 0
        return cost / Flat.max_cost

    @staticmethod
    def decode_cost(cost: float) -> float:
        assert Flat.max_cost > 0
        return cost * Flat.max_cost

    def __str__(self) -> str:
        return "\n".join([
            "Type: " + self.flat_type,
            "City: " + self.city,
            "Street: " + self.street,
            "Cost: %5d Area: %5d" % (self.cost, self.area)
        ])


def train(training_data: list) -> NeuralNetwork:

    nn = NeuralNetwork.generate(3, [3, 3, 1])

    for i in range(100):

        nn.train(100, training_data, 0.1, 10)

        print("MSE = %lf, Average Error = %lf" % run_test(nn, training_data, True))

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


def run_test(nn: NeuralNetwork, data: list, silent=False) -> (float, float):

    mse = 0
    error_avg = 0

    for inp, out in data:
        predicted = nn.run(inp)

        error = out[0][0] - predicted[0][0]

        mse += error * error

        error_avg += abs(out[0][0] - predicted[0][0])

        if silent:
            continue

        print("Expected: %lf Predicted: %lf (%lf Euro) Error: %lf MSE: %lf" % (
            out[0][0],
            predicted[0][0],
            Flat.decode_cost(predicted[0][0]),
            abs(error),
            mse
        ))

    error_avg /= len(data)

    if not silent:
        print("MSE: %lf" % mse)
        print("Average Error: %lf (%lf Euro)" % (error_avg, Flat.decode_cost(error_avg)))

    return mse, error_avg
