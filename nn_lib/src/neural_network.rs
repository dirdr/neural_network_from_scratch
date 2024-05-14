use ndarray::Array2;

use crate::layer::Layer;

pub struct NeuralNetworkBuilder {
    layers: Vec<Box<dyn Layer>>,
    learning_rate: f64,
    epochs: usize,
    gradient_descent_strategy: GradientDescentStrategy,
}

pub enum NeuralNetworkError {
    MissingMandatoryFields(String),
}

impl NeuralNetworkBuilder {
    /// Create a new `NeuralNetorkBuilder` with the following default values :
    /// * `learning_rate`: 0.1
    /// * `epochs`: 0.1
    /// * `gradient_descent_strategy`: MiniBatch
    pub fn new() -> Self {
        Self {
            layers: vec![],
            learning_rate: 0.1,
            epochs: 1000,
            gradient_descent_strategy: GradientDescentStrategy::MiniBatch,
        }
    }

    pub fn build(self) -> Result<NeuralNetwork, NeuralNetworkError> {
        Ok(NeuralNetwork {
            layers: self.layers,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
        })
    }

    pub fn add_input_layer(mut self, input_sizej te)

    /// Add a layer to the sequential neural network
    /// in a sequential neural network, layers are added left to right (input -> hidden -> output)
    pub fn add_layer(mut self, layer: Box<dyn Layer>) -> Self {
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

    pub fn gradient_descent_strategy(mut self, gds: GradientDescentStrategy) -> Self {
        self.gradient_descent_strategy = gds;
        self
    }
}

impl Default for NeuralNetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Enumeration of Gradient Descent Strategy
/// `Batch` use the whole dataset to make a gradient descent step
/// `MiniBatch` use randomly shuffled subset of the original input
/// `Stochastic` use a random data point of the original set
pub enum GradientDescentStrategy {
    Batch,
    MiniBatch,
    Stochastic,
}

/// a trainable `NeuralNetwork`
/// # Fields
/// * `layers` - A vector of layers (could be activation, convolutional, dense, etc..) in
/// sequencial order
/// * `epochs` - number of time the whole dataset will be used to train the network
/// * `learning_rate` - gradient descent learning rate
pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
    epochs: usize,
    learning_rate: f64,
}

impl NeuralNetwork {
    fn predict(&self, input: Array2<f64>) -> Array2<f64> {
        todo!()
    }

    /// Train the neural network
    fn train(&self) {}
}
