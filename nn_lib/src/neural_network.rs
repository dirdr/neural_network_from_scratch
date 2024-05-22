use crate::{
    cost::CostFunction,
    layer::{DenseLayer, Layer, LayerError},
    optimizer::Optimizer,
};
use log::debug;
use ndarray::{ArrayD, Axis};
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;
use thiserror::Error;

pub struct NeuralNetworkBuilder {
    layers: Vec<Box<dyn Layer>>,
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
    pub fn new() -> NeuralNetworkBuilder {
        Self { layers: vec![] }
    }

    /// Add a layer to the sequential neural network
    /// in a sequential neural network, layers are added left to right (input -> hidden -> output)
    pub fn push(mut self, layer: impl Layer + 'static) -> NeuralNetworkBuilder {
        self.layers.push(Box::new(layer));
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
    layers: Vec<Box<dyn Layer>>,
    cost_function: CostFunction,
    optimizer: Box<dyn Optimizer>,
}

impl NeuralNetwork {
    /// predict from the neural network
    /// the shape of the prediction is (n, dim o) where **dim o** is the dimension of the network
    /// last layer and **n** is the number of point in the batch.
    ///
    /// # Arguments
    /// * `input` : batched input, of size (n, dim i) where **dim i** is the dimension of the
    /// network first layer and **n** is the number of point in the batch.
    pub fn predict(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.feed_forward(&output)?;
        }
        Ok(output)
    }

    /// Train the neural network with Gradient descent Algorithm
    /// # Arguments
    /// * `x_train` - an ArrayD of which the outer dimension need to contains the data, for exemple
    /// for a collections of 60000 images of 28*28, the ArrayD dimension will be [60000, 28, 28].
    /// * `y_train` - an ArrayD of which the outer dimension need to contains the data like x_train
    /// one-hot encoded.
    pub fn train(
        &mut self,
        x_train: ArrayD<f64>,
        y_train: ArrayD<f64>,
        epochs: usize,
        batch_size: usize,
    ) -> Result<(), LayerError> {
        for e in 0..epochs {
            debug!("Inside epochs {}", e);
            assert!(x_train.shape()[0] == y_train.shape()[0]);

            let mut indices = (0..x_train.shape()[0]).collect::<Vec<_>>();
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);

            indices.chunks(batch_size).try_for_each(|batch_i| {
                let batched_x = x_train.select(Axis(0), batch_i);
                // TODO vérifier si ca ca marche bien, et soit mieux spécifier dans la doc le
                // format de données attendu, soit faire des check avant de reshape volontairement
                let batched_y = y_train.select(Axis(0), batch_i);
                self.process_batch(batched_x, batched_y)?;
                Ok::<(), LayerError>(())
            })?;
        }
        Ok(())
    }

    pub fn process_batch(
        &mut self,
        batched_x: ArrayD<f64>,
        batched_y: ArrayD<f64>,
    ) -> Result<(), LayerError> {
        // debug!("batched x shape : {:?}", batched_x.shape());
        // debug!("batched y shape : {:?}", batched_y.shape());
        let output = self.predict(&batched_x)?;
        //let cost = self.cost_function.cost(&output, &batched_y);
        self.backpropagation(output, batched_y)?;
        Ok(())
    }

    fn backpropagation(
        &mut self,
        net_output: ArrayD<f64>,
        observed: ArrayD<f64>,
    ) -> Result<(), LayerError> {
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

        for layer in self.layers.iter_mut().rev().skip(skip_layer) {
            grad = layer.propagate_backward(&grad)?;

            // Downcast to Trainable and call optimizer's step method if possible
            // if other layers (like convolutional implement trainable, need to downcast
            // explicitely)
            if let Some(trainable_layer) = layer.as_any_mut().downcast_mut::<DenseLayer>() {
                self.optimizer.step(trainable_layer);
            }
        }
        Ok(())
    }
}
