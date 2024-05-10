# About

This simple project aims to create a simple neural network from scratch, and train it with the MNIST database, to make him recongnise handwritten digits.

# Usage

## Performance benchmark

The project can be run to benchmark his performance, both **with and without convolution**,
benchamrking is done in learning, and recognition.

## Interactive usage

You can also play with an interactive gui, drawing your own number and see what the trained model guess.

# Dataset

the dataset used is the **MNIST Databse** of handwritten digits containing a training set of 60k samples, and a testing set of 10k samples.\
the files are stored inside the [resources folder](./resources/), there is 4 files

- [training set images](./resources/train-images-idx3-ubyte.gz)
- [training set labels](./resources/train-images-idx1-ubyte.gz)
- [test set images](./resources/t10k-images-idx3-ubyte.gz)
- [test set labels](./resources/train-images-idx1-ubyte.gz)

  At the time we wrote the program, the files in the [official respository](http://yann.lecun.com/exdb/mnist/) where unavaible, we downloaded the version from [this mirror](https://github.com/mkolod/MNIST).
