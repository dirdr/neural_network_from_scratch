use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use crate::{cost::CostFunction, layer::Layer};
use log::info;
use ndarray::{par_azip, Array2, Array3, Axis};

pub struct NeuralNetworkBuilder {
    layers: Vec<Arc<Mutex<dyn Layer>>>,
    learning_rate: f64,
    epochs: usize,
    gradient_descent_strategy: GradientDescentStrategy,
    cost_function: CostFunction,
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
/// sequencial order
/// * `epochs` - number of time the whole dataset will be used to train the network
/// * `learning_rate` - gradient descent learning rate
pub struct NeuralNetwork {
    layers: Vec<Arc<Mutex<dyn Layer>>>,
    epochs: usize,
    learning_rate: f64,
    cost_function: CostFunction,
}

impl NeuralNetwork {
    /// Train the neural network with Gradient descent algorithm
    /// # Arguments
    /// * `x_train` - a Array3 (shape (num_train_samples, i, n)) of training images
    /// * `y_train` - a Array3 (shape (num_label_samples, j, 1)) of training label labels are
    /// one-hot encoded.
    /// * `cost` - cost function used to calcualte the error magnitude.
    pub fn train(&mut self, x_train: Array3<f64>, y_train: Array3<f64>) {
        let output_shape = y_train.index_axis(Axis(0), 0).raw_dim();
        let layers = self.layers.clone();
        let learning_rate = self.learning_rate;
        let cost_function = self.cost_function;
        let epochs = self.epochs;
        for e in 0..epochs {
            let error = Arc::new(Mutex::new(Array2::zeros(output_shape)));
            info!("Successfully passed through an epoch");
            let count = Arc::new(AtomicUsize::new(0));
            par_azip!((x in x_train.outer_iter(), y in y_train.outer_iter()) {
                let (x, y) = (x.to_owned(), y.to_owned());

                // feed forward
                let output = {
                    let mut output = x.clone();
                    for layer in &layers {
                        let mut layer = layer.lock().unwrap();
                        output = layer.feed_forward(&output);
                    }
                    output
                };

                // cost evaluation
                let cost = cost_function.cost(&output, &y);
                {
                    let mut error_guard = error.lock().unwrap();
                    *error_guard += cost;
                }

                // first cost function gradient
                let mut grad = cost_function.cost_output_gradient(&y, &output);

                // back propagation (weight and bias update)
                for layer in layers.iter().rev() {
                    let mut layer = layer.lock().unwrap();
                    grad = layer.propagate_backward(&grad, learning_rate);
                }

                let index = count.fetch_add(1, Ordering::SeqCst);
                info!("Processing training sample {}", index);
            });

            let error = Arc::try_unwrap(error).unwrap().into_inner().unwrap();
            let error = error / x_train.len() as f64;
            info!("Epochs : {}, training_error : {}", e, error);
        }
    }
}
