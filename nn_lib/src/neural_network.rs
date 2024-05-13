use ndarray::Array2;

use crate::layer::Layer;

struct NeuralNetworkBuilder<L>
where
    L: Layer,
{
    layers: Vec<L>,
    learning_rate: f64,
    epochs: usize,
}

impl<L> NeuralNetworkBuilder<L>
where
    L: Layer,
{
    /// Create a new `NeuralNetorkBuilder` with default values
    pub fn new() -> Self {
        Self {
            layers: vec![],
            learning_rate: 0.1,
            epochs: 1000,
        }
    }

    pub fn build(self) -> NeuralNetwork<L> {
        NeuralNetwork {
            layers: self.layers,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
        }
    }

    /// Add a layer to the sequential neural network
    /// in a sequential neural network, layers are added left to right (input -> hidden -> output)
    pub fn add_layer(mut self, layer: L) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }
}

struct NeuralNetwork<L>
where
    L: Layer,
{
    layers: Vec<L>,
    epochs: usize,
    learning_rate: f64,
}

impl<L> NeuralNetwork<L>
where
    L: Layer,
{
    fn predict(&self, input: Array2<f64>) -> Array2<f64> {
        todo!()
    }

    fn train(&self) {
        todo!()
    }
}
