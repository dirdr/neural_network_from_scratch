use ndarray::Array2;

use crate::layer::Layer;

struct NeuralNetworkBuilder<L>
where
    L: Layer,
{
    layers: Vec<L>,
    learning_rate: f64,
    epochs: usize,
    gradient_descent_strategy: GradientDescentStrategy,
}

pub enum NeuralNetworkError {
    MissingMandatoryFields(String),
}

impl<L> NeuralNetworkBuilder<L>
where
    L: Layer,
{
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

    pub fn build(self) -> Result<NeuralNetwork<L>, NeuralNetworkError> {
        Ok(NeuralNetwork {
            layers: self.layers,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
        })
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

    pub fn gradient_descent_strategy(mut self, gds: GradientDescentStrategy) -> Self {
        self.gradient_descent_strategy = gds;
        self
    }
}

/// Enumeration of Gradient Descent Strategy
/// `Batch` use the whole dataset to make a gradient descent step
/// `MiniBatch` use randomly shuffled subset of the original input
/// `Stochastic` use a random data point of the original set
enum GradientDescentStrategy {
    Batch,
    MiniBatch,
    Stochastic,
}

/// a trainable `NeuralNetwork`
/// # Fields
/// * `layers` - A vector of layers (could be activation, convolutional, dense, etc..) in
/// sequencial order
/// * `epochs`
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

    /// Train the neural network
    fn train(&self) {}
}
