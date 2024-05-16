use crate::{cost::Cost, layer::Layer};
use log::info;
use ndarray::{Array2, Array3, Axis, NdProducer};

pub struct NeuralNetworkBuilder {
    layers: Vec<Box<dyn Layer>>,
    learning_rate: f64,
    epochs: usize,
    gradient_descent_strategy: GradientDescentStrategy,
    cost_function: Cost,
}

pub enum NeuralNetworkError {
    MissingMandatoryFields(String),
}

impl NeuralNetworkBuilder {
    /// Create a new `NeuralNetorkBuilder` with the following default values :
    /// * `learning_rate`: 0.1
    /// * `epochs`: 0.1
    /// * `gradient_descent_strategy`: MiniBatch
    pub fn new() -> NeuralNetworkBuilder {
        Self {
            layers: vec![],
            learning_rate: 0.1,
            epochs: 1000,
            gradient_descent_strategy: GradientDescentStrategy::MiniBatch,
            cost_function: Cost::CrossEntropy,
        }
    }
}

impl NeuralNetworkBuilder {
    /// Add a layer to the sequential neural network
    /// in a sequential neural network, layers are added left to right (input -> hidden -> output)
    pub fn push_layer(mut self, layer: Box<dyn Layer>) -> NeuralNetworkBuilder {
        self.layers.push(layer);
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> NeuralNetworkBuilder {
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

    pub fn build(self) -> Result<NeuralNetwork, NeuralNetworkError> {
        Ok(NeuralNetwork {
            layers: self.layers,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
        })
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
    pub fn predict(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.feed_forward(&output);
        }
        output
    }

    /// Train the neural network
    pub fn train(&mut self, x_train: Array3<f64>, y_train: Array3<f64>, cost: Cost) {
        let output_shape = y_train.index_axis(Axis(0), 0).raw_dim();
        for e in 0..self.epochs {
            let mut error: Array2<f64> = Array2::zeros(output_shape);

            // TODO handle multiple Gradient descent strategy
            // TODO wrap function inside a GradientDescent method
            for (x, y) in x_train.iter().zip(y_train.iter()) {
                let output = &self.predict(x);

                error += cost.cost(y, output);

                let mut grad = cost.cost_output_gradient(y, output);

                for layer in self.layers.iter_mut().rev() {
                    grad = layer.propagate_backward(&grad, self.learning_rate);
                }
            }

            error /= x_train.len() as f64;
            info!("Epochs : {}, training_error : {}", e, error)
        }
    }
}
