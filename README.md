# About
This repository hold multiple crates, among them [./nn_lib] which is the main neural network library.

Before we decided to improve the neural network library by adding more features, this was a school project,
you check out the [report](./report/nn_from_scratch.pdf) (pdf format) we wrote explaining the basic structure of the library and the maths behind our implementation.
The [latex sources](./report/) of the report is also avaible.

This application, using our library, serve as an entrypoint for our school project which was to solve the mnist dataset.
There are two mode, `benchmark` and `gui`, the first one give metrics and loss for either mnist or xor, and the second one is a drawing GUI around the mnist dataset.

# Exemple
Launch the mnist gui with data augmentation:
```sh
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

## Network performance benchmark
The project can be run to benchmark the network performances, see the help command

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
You can also play with an interactive gui for the mnist exemple, drawing your own number and see what the trained model guess.

```txt
Run in GUI mode

Usage: nn_from_scratch gui [OPTIONS]

Options:
  -a, --augment
  -h, --help     Print help
```
