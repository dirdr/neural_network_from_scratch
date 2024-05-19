use crate::{
    cost::CostFunction,
    layer::{DenseLayer, Layer},
    optimizer::Optimizer,
};
use log::{debug, info};
use ndarray::{par_azip, Array2, Array3};
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use std::{
    os::unix::thread,
    sync::{Arc, Mutex},
};
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
            optimizer: Arc::new(Mutex::new(optimizer)),
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
    optimizer: Arc<Mutex<dyn Optimizer>>,
}

impl NeuralNetwork {
    /// predict from the neural network
    /// the shape of the prediction is defined by the neural net's last layer shape.
    /// # Arguments
    /// * `input` : the input of the neural network.
    pub fn predict(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = input.clone();
        for layer in &self.layers {
            let mut layer = layer.lock().unwrap();
            output = layer.feed_forward(&output);
        }
        output
    }

    /// Train the neural network with Gradient descent Algorithm
    /// # Arguments
    /// * `x_train` - a vec of Array2 (shape (i, n)) of training images
    /// * `y_train` - a vec of Array2 (shape (j, 1)) of training label labels are
    /// one-hot encoded.
    pub fn train_par(
        &mut self,
        x_train: Vec<Array2<f64>>,
        y_train: Vec<Array2<f64>>,
        epochs: usize,
        batch_size: usize,
    ) {
        for _ in 0..epochs {
            assert!(x_train.len() == y_train.len());

            let mut indices = (0..x_train.len()).collect::<Vec<_>>();
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);

            indices.chunks(batch_size).for_each(|batch_i| {
                //debug!("indices batches {:?} for epochs {}", batch_i, e);
                let batched_x: Vec<Array2<f64>> =
                    batch_i.iter().map(|&i| x_train[i].clone()).collect();
                let batched_y: Vec<Array2<f64>> =
                    batch_i.iter().map(|&i| y_train[i].clone()).collect();
                self.process_batch(batched_x, batched_y);
            });
        }
    }

    pub fn process_batch(&mut self, batched_x: Vec<Array2<f64>>, batched_y: Vec<Array2<f64>>) {
        let error = Arc::new(Mutex::new(0.0));
        batched_x
            .par_iter()
            .zip(batched_y.par_iter())
            .for_each(|(x, y)| {
                let (x, y) = (x.to_owned(), y.to_owned());

                let output = self.predict(&x);

                // Cost evaluation
                let cost = self.cost_function.cost(&output, &y);

                // Update error
                {
                    let mut error_guard = error.lock().unwrap();
                    *error_guard += cost;
                }

                self.backpropagation(output, y);
            });
        //let error = Arc::try_unwrap(error).unwrap().into_inner().unwrap() / batched_x.len() as f64;
    }

    fn backpropagation(&self, net_output: Array2<f64>, observed: Array2<f64>) {
        let mut grad = self
            .cost_function
            .cost_output_gradient(&net_output, &observed);
        // if the cost function is dependant of the last layer, the gradient calculation
        // have been done with respect to the net logits directly, thus skip the last layer
        // in the gradients backpropagation
        let skip_layer = if self.cost_function.output_dependant() {
            1
        } else {
            0
        };

        for layer in self.layers.iter().rev().skip(skip_layer) {
            let mut layer = layer.lock().unwrap();
            grad = layer.propagate_backward(&grad);

            // Downcast to Trainable and call optimizer's step method if possible
            // if other layers (like convolutional implement trainable, need to downcast
            // explicitely)
            if let Some(trainable_layer) = layer.as_any_mut().downcast_mut::<DenseLayer>() {
                let mut optimizer = self.optimizer.lock().unwrap();
                optimizer.step(trainable_layer);
            }
        }
    }
}
