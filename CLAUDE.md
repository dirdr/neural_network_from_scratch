# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a Rust-based neural network implementation from scratch with three main components:

- **`nn_lib/`** - Core neural network library with layers, activations, optimizers, and training logic
- **`mnist/`** - MNIST dataset handling, network definitions, and training routines  
- **`src/`** - Main application with CLI interface supporting GUI and benchmark modes

The project uses a workspace structure with `nn_lib` and `mnist` as separate crates.

## Core Architecture

The neural network is built using a **Sequential** model pattern from `nn_lib/src/sequential.rs`. Networks are constructed using `SequentialBuilder` by chaining layers:

- **Dense layers** for fully connected neurons
- **Convolutional layers** for feature extraction
- **Activation layers** (ReLU, Softmax, etc.)
- **Pooling layers** (MaxPooling)
- **Reshape layers** for tensor transformations

Two network types are implemented in `mnist/src/network_definition.rs`:
- **MLP (Multi-Layer Perceptron)** - Basic feedforward network
- **CNN (Convolutional Neural Network)** - With conv + pooling layers

## Development Commands

### Build and Run
```bash
# Build in release mode (recommended for training)
cargo build --release

# Run GUI mode with data augmentation
RUST_LOG=trace cargo run --release -- gui --augment

# Run MNIST benchmark with MLP
cargo run --release -- benchmark --run mnist --net-type mlp

# Run MNIST benchmark with CNN
cargo run --release -- benchmark --run mnist --net-type conv

# Run XOR benchmark
cargo run --release -- benchmark --run xor
```

### Available CLI Options
- **GUI mode**: Interactive drawing interface for MNIST digit recognition
  - `--augment`: Enable data augmentation
- **Benchmark mode**: Training and evaluation metrics
  - `--run`: Choose dataset (mnist, xor)
  - `--net-type`: Choose network architecture (mlp, conv)  
  - `--epochs`: Set training epochs

### Logging
Use `RUST_LOG=trace` environment variable for detailed logging during training and inference.

## Key Implementation Details

- Networks use **ndarray** for tensor operations with optional BLAS acceleration
- Training supports **batch processing** (default batch size: 128)  
- **Metrics tracking** system for accuracy, loss, and custom metrics
- **Data augmentation** available for MNIST dataset
- **GUI framework** uses egui/eframe for the interactive drawing interface
- **MNIST data** stored in `mnist/resources/` (both compressed and raw formats)

The library implements gradient descent optimization with support for different weight initializers (He, Glorot/Xavier) and cost functions.