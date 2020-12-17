# FLAI
AI that analyzes flat/apartment offers and predicts prices, neural network implemented in pure python &amp; numpy.

## Usage

Follow these steps to setup FLAI:

1. Clone this repository with `git clone https://github.com/ZeroBone/FLAI.git`.
2. Create a `trained` directory where the trained neural networks will be stored.
3. Add your dataset to the `data` directory or use one of the ready datasets.

### Training and testing the neural network

To train the neural network, run `python train.py <dataset>` where `<dataset>` is the name of the csv file with the data to be used for training. Specify the dataset name without `.csv`.

The network can be tested analogously, by executing `python test.py <dataset>`.

