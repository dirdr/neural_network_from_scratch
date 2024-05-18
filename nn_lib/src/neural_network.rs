use std::sync::{Arc, Mutex};

use crate::{cost::CostFunction, layer::Layer, optimizer::Optimizer};
use log::info;
use ndarray::{par_azip, Array2, Array3};
use thiserror::Error;

pub struct NeuralNetworkBuilder {
    layers: Vec<Arc<Mutex<dyn Layer>>>,
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
    /// Create a new `NeuralNetworkBuilder`
    pub fn new() -> NeuralNetworkBuilder {
        Self { layers: vec![] }
    }
}

impl NeuralNetworkBuilder {
    /// Add a layer to the sequential neural network
    /// in a sequential neural network, layers are added left to right (input -> hidden -> output)
    pub fn push(mut self, layer: impl Layer + 'static) -> NeuralNetworkBuilder {
        self.layers.push(Arc::new(Mutex::new(layer)));
        self
    }

    /// Build the neural network
    /// A NeuralNetworkError is returned if the network is wrongly defined
    /// see `NeuralNetworkError` for informations of what can fail.
    pub fn build(
        self,
        optimizer: impl Optimizer + 'static,
        cost_function: CostFunction,
    ) -> Result<NeuralNetwork, NeuralNetworkError> {
        // TODO check if the cost function and last layer match
        // TODO check of the network dimension are ok
        Ok(NeuralNetwork {
            layers: self.layers,
            cost_function,
            optimizer: Box::new(optimizer),
        })
    }
}

impl Default for NeuralNetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
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
/// * cost_function - TODO
/// * optimoizer - TODO
pub struct NeuralNetwork {
    layers: Vec<Arc<Mutex<dyn Layer>>>,
    cost_function: CostFunction,
    optimizer: Box<dyn Optimizer>,
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
    pub fn train_par(&mut self, x_train: Array3<f64>, y_train: Array3<f64>, epochs: usize) {
        let layers = self.layers.clone();
        let cost_function = self.cost_function;
        let learning_rate = self.optimizer.get_learning_rate();
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
                    grad = layer.propagate_backward(&grad);
                }
            });

            let error =
                Arc::try_unwrap(error).unwrap().into_inner().unwrap() / x_train.len() as f64;
            info!("Epoch {}: training error = {}", e, error);
        }
    }
}
