use std::sync::{Arc, Mutex};

use crate::{cost::CostFunction, layer::Layer};
use log::info;
use ndarray::{par_azip, Array2, Array3};
use thiserror::Error;

pub struct NeuralNetworkBuilder {
    layers: Vec<Arc<Mutex<dyn Layer>>>,
    learning_rate: f64,
    epochs: usize,
    gradient_descent_strategy: GradientDescentStrategy,
    cost_function: CostFunction,
}

#[derive(Error, Debug)]
pub enum NeuralNetworkError {
    #[error("Missing mandatory fields to build the network")]
    MissingMandatoryFields,

    #[error(
        "Invalid output activation layer,
        see CostFunction::output_dependant for detailed explanation. provided : {0}"
    )]
    WrongOutputActivationLayer(String),
}

impl NeuralNetworkBuilder {
    /// Create a new `NeuralNetworkBuilder` with the following default values :
    /// * `learning_rate`: 0.1
    /// * `epochs`: 0.1
    /// * `gradient_descent_strategy`: MiniBatch
    pub fn new() -> NeuralNetworkBuilder {
        Self {
            layers: vec![],
            learning_rate: 0.1,
            epochs: 1000,
            gradient_descent_strategy: GradientDescentStrategy::MiniBatch,
            cost_function: CostFunction::CrossEntropy,
        }
    }
}

impl NeuralNetworkBuilder {
    /// Add a layer to the sequential neural network
    /// in a sequential neural network, layers are added left to right (input -> hidden -> output)
    pub fn push_layer(mut self, layer: impl Layer + 'static) -> NeuralNetworkBuilder {
        self.layers.push(Arc::new(Mutex::new(layer)));
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

    pub fn with_cost_function(mut self, cost_function: CostFunction) -> Self {
        self.cost_function = cost_function;
        self
    }

    pub fn build(self) -> Result<NeuralNetwork, NeuralNetworkError> {
        Ok(NeuralNetwork {
            layers: self.layers,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            cost_function: self.cost_function,
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
/// sequential order
/// note that this crate dont use autodijff, so if you are planning to use a neural net architecture
/// with cross entropy, or binary cross entropy, the network make and use the assumption of
/// softmax, and sigmoid activation function respectively just before the cost function.
/// Thus you don't need to include it in the layers. However if you use any kind of independant
/// cost function (like mse) you can include whatever activation function you wan't after the
/// output because the gradient calculation is independant of the last layer you choose.
/// * `epochs` - number of time the whole dataset will be used to train the network
/// * `learning_rate` - gradient descent learning rate
pub struct NeuralNetwork {
    layers: Vec<Arc<Mutex<dyn Layer>>>,
    epochs: usize,
    learning_rate: f64,
    cost_function: CostFunction,
}

impl NeuralNetwork {
    /// predict from the neural network
    /// the shape of the prediction is defined by the neural net's last layer shape.
    /// # Arguments
    /// * `input` : the input of the neural network.
    pub fn predict(&self, input: Array2<f64>) -> Array2<f64> {
        let mut output = input.clone();
        for layer in &self.layers {
            let mut layer = layer.lock().unwrap();
            output = layer.feed_forward(&output);
        }
        output
    }

    /// Train the neural network with Gradient descent algorithm
    /// # Arguments
    /// * `x_train` - an Array3 (shape (num_train_samples, i, n)) of training images
    /// * `y_train` - an Array3 (shape (num_label_samples, j, 1)) of training label labels are
    /// one-hot encoded.
    pub fn train_par(&mut self, x_train: Array3<f64>, y_train: Array3<f64>) {
        let layers = self.layers.clone();
        let learning_rate = self.learning_rate;
        let cost_function = self.cost_function;
        let epochs = self.epochs;

        for e in 0..epochs {
            let error = Arc::new(Mutex::new(0.0));

            par_azip!((x in x_train.outer_iter(), y in y_train.outer_iter()) {
                let (x, y) = (x.to_owned(), y.to_owned());


                // Feed forward
                let output = {
                    let mut output = x.clone();
                    for layer in &layers {
                        let mut layer = layer.lock().unwrap();
                        output = layer.feed_forward(&output);
                    }
                    output
                };

                // Cost evaluation
                let cost = cost_function.cost(&output, &y);
                {
                    let mut error_guard = error.lock().unwrap();
                    *error_guard += cost;
                }

                // First cost function gradient
                let mut grad = cost_function.cost_output_gradient(&output, &y);

                // Back propagation (weight and bias update)
                for layer in layers.iter().rev() {
                    let mut layer = layer.lock().unwrap();
                    grad = layer.propagate_backward(&grad, learning_rate);
                }
            });

            let error =
                Arc::try_unwrap(error).unwrap().into_inner().unwrap() / x_train.len() as f64;
            info!("Epoch {}: training error = {}", e, error);
        }
    }
}
