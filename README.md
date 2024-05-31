# About
This simple project aims to create a simple neural network from scratch, and train it with the MNIST database, to make him recongnise handwritten digits.

# Exemple
Launch the gui with data augmentation:
```s
RUST_LOG=trace cargo run --release -- gui --augment
```

# Usage
```txt
A simple neural network library written in rust

Usage: nn_from_scratch <COMMAND>

Commands:
  gui        Run in GUI mode
  benchmark  Run benchmarks
  help       Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
 ```

## Performance benchmark
The project can be run to benchmark performance, see the help command

```txt
Run benchmarks

Usage: nn_from_scratch benchmark [OPTIONS]

Options:
  -r, --run <RUN>            [default: xor] [possible values: mnist, xor]
  -e, --epochs <EPOCHS>
  -n, --net-type <NET_TYPE>  [default: mlp] [possible values: mlp, conv]
  -h, --help                 Print help
```

## Interactive usage
You can also play with an interactive gui, drawing your own number and see what the trained model guess.

```txt
Run in GUI mode

Usage: nn_from_scratch gui [OPTIONS]

Options:
  -a, --augment
  -h, --help     Print help
```

# Dataset

the dataset used is the **MNIST Databse** of handwritten digits containing a training set of 60k samples, and a testing set of 10k samples.\
the files are stored inside the [resources folder](./resources/), there is 4 files

- [training set images](./resources/train-images-idx3-ubyte.gz)
- [training set labels](./resources/train-images-idx1-ubyte.gz)
- [test set images](./resources/t10k-images-idx3-ubyte.gz)
- [test set labels](./resources/train-images-idx1-ubyte.gz)

  At the time we wrote the program, the files in the [official respository](http://yann.lecun.com/exdb/mnist/) where unavaible, we downloaded the version from [this mirror](https://github.com/mkolod/MNIST).
