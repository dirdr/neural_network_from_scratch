use crate::{
    activation::Activation,
    cost::CostFunction,
    layer::{ActivationLayer, DenseLayer, Layer, LayerError},
    metrics::{Benchmark, History, MetricsType},
    optimizer::Optimizer,
};
use log::debug;
use ndarray::{ArrayD, Axis};
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;
use thiserror::Error;

pub struct NeuralNetworkBuilder {
    layers: Vec<Box<dyn Layer>>,
    metrics: Vec<MetricsType>,
}

impl NeuralNetworkBuilder {
    pub fn new() -> NeuralNetworkBuilder {
        Self {
            layers: vec![],
            metrics: vec![],
        }
    }

    /// Add a layer to the sequential neural network
    /// in a sequential neural network, layers are added left to right (input -> hidden -> output)
    pub fn push(mut self, layer: impl Layer + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Add a metric to compute for the neural network,
    /// added metrics will be avaible inside the history record and inside the bench object that
    /// the method evaluate return
    pub fn watch(mut self, metric_type: MetricsType) -> Self {
        self.metrics.push(metric_type);
        self
    }

    /// Add all the metric inside `metrics` into the neural network metrics watch list
    pub fn watch_all(mut self, metrics: Vec<MetricsType>) -> Self {
        self.metrics.extend(metrics.iter());
        self
    }

    /// Build the neural network.
    /// Returns a `NeuralNetworkError` if the network is wrongly defined.
    /// See `NeuralNetworkError` for information on what can fail.
    pub fn compile(
        self,
        optimizer: impl Optimizer + 'static,
        cost_function: CostFunction,
    ) -> Result<NeuralNetwork, NeuralNetworkError> {
        // Check if the cost function is compatible with the last layer's activation function
        if cost_function.is_output_dependant() {
            self.validate_last_layer_activation(&cost_function)?;
        }

        Ok(NeuralNetwork {
            layers: self.layers,
            cost_function,
            optimizer: Box::new(optimizer),
            metrics: self.metrics,
        })
    }

    /// Validates that the last layer's activation function is compatible with the given cost function.
    fn validate_last_layer_activation(
        &self,
        cost_function: &CostFunction,
    ) -> Result<(), NeuralNetworkError> {
        self.layers
            .last()
            .and_then(|layer| layer.as_any().downcast_ref::<ActivationLayer>())
            .map_or(
                Err(NeuralNetworkError::MissingActivationLayer),
                |activation_layer| match cost_function {
                    CostFunction::Mse => Ok(()),
                    CostFunction::CrossEntropy
                        if activation_layer.activation == Activation::Softmax =>
                    {
                        Ok(())
                    }
                    CostFunction::BinaryCrossEntropy
                        if activation_layer.activation == Activation::Sigmoid =>
                    {
                        Ok(())
                    }
                    _ => Err(NeuralNetworkError::WrongOutputActivationLayer),
                },
            )
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
    metrics: Vec<MetricsType>,
}

impl NeuralNetwork {
    /// predict a value from the neural network
    /// the shape of the prediction is (n, dim o) where **dim o** is the dimension of the network
    /// last layer and **n** is the number of point in the batch.
    ///
    /// # Arguments
    /// * `input` : batched input, of size (n, dim i) where **dim i** is the dimension of the
    /// network first layer and **n** is the number of point in the batch.
    pub fn predict(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.feed_forward(&output)?;
        }
        Ok(output)
    }

    /// Evaluate the **trained** neural network on a test input and observed values.
    /// returning a `Benchmark` containing the error on the test set, along with the metrics
    /// provided
    ///
    /// # Arguments
    /// * `x_test` test data set, the outer dimension must contains the data
    /// * `y_test` test observed values, the outer dimension must contains the data
    /// * `metrics` optional metrics struct
    /// * `batch_size` the batch size, ie: number of data point treaded simultaneously
    pub fn evaluate(&self, x: &ArrayD<f64>, y: &ArrayD<f64>, batch_size: usize) -> Benchmark {
        let mut bench = Benchmark::new(&self.metrics);
        assert!(x.shape()[0] == y.shape()[0]);
        let batches = Self::create_batches(x, y, batch_size);

        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for (batched_x, batched_y) in batches.into_iter() {
            let output = self.predict(&batched_x).unwrap();

            let batch_loss = self.cost_function.cost(&output, &batched_y);

            if !self.metrics.is_empty() {
                bench.metrics.accumulate(&output, &batched_y);
            }

            total_loss += batch_loss;
            batch_count += 1;
        }

        bench.metrics.mean_all(batch_count);
        bench.loss = total_loss / batch_count as f64;
        bench
    }

    /// Train the neural network with Gradient descent Algorithm
    /// # Arguments
    /// * `train_data`
    pub fn train(
        &mut self,
        train_data: (&ArrayD<f64>, &ArrayD<f64>),
        validation_data: Option<(&ArrayD<f64>, &ArrayD<f64>)>,
        epochs: usize,
        batch_size: usize,
    ) -> Result<(History, Option<History>), LayerError> {
        let (x_train, y_train) = train_data;

        if x_train.shape()[0] != y_train.shape()[0] {
            return Err(LayerError::DimensionMismatch);
        }

        let mut train_history = History::new();
        let mut validation_history = validation_data.map(|_| History::new());

        let batches = Self::create_batches(x_train, y_train, batch_size);

        for e in 0..epochs {
            debug!("Training epochs : {}", e);
            let epoch_result = self.process_epoch(&batches)?;
            train_history.history.push(epoch_result);

            if let Some((x_val, y_val)) = validation_data {
                let validation_bench = self.evaluate(x_val, y_val, batch_size);
                validation_history
                    .as_mut()
                    .unwrap()
                    .history
                    .push(validation_bench);
            }
        }

        Ok((train_history, validation_history))
    }

    fn process_epoch(
        &mut self,
        batches: &[(ArrayD<f64>, ArrayD<f64>)],
    ) -> Result<Benchmark, LayerError> {
        let mut bench = Benchmark::new(&self.metrics);
        let mut total_loss = 0.0;

        for (batched_x, batched_y) in batches.iter() {
            let output = self.feed_forward(batched_x)?;
            let batch_loss = self.cost_function.cost(&output, batched_y);

            // the cost function is already meaned over the data point of the batch
            total_loss += batch_loss;

            bench.metrics.accumulate(&output, batched_y);
            self.backpropagation(&output, batched_y)?;
        }

        bench.metrics.mean_all(batches.len());
        bench.loss = total_loss / batches.len() as f64;

        Ok(bench)
    }

    fn create_batches(
        x_train: &ArrayD<f64>,
        y_train: &ArrayD<f64>,
        batch_size: usize,
    ) -> Vec<(ArrayD<f64>, ArrayD<f64>)> {
        let mut indices = (0..x_train.shape()[0]).collect::<Vec<_>>();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
        indices
            .chunks(batch_size)
            .map(|batch_indices| {
                (
                    x_train.select(Axis(0), batch_indices),
                    y_train.select(Axis(0), batch_indices),
                )
            })
            .collect::<Vec<_>>()
    }

    pub fn feed_forward(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.feed_forward_save(&output)?;
        }
        Ok(output)
    }

    fn backpropagation(
        &mut self,
        net_output: &ArrayD<f64>,
        observed: &ArrayD<f64>,
    ) -> Result<(), LayerError> {
        let mut grad = self
            .cost_function
            .cost_output_gradient(&net_output, &observed);

        // if the cost function is dependant of the last layer, the gradient calculation
        // have been done with respect to the net logits directly, thus skip the last layer
        // in the gradients backpropagation
        let skip_layer = if self.cost_function.is_output_dependant() {
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

#[derive(Error, Debug)]
pub enum NeuralNetworkError {
    #[error("Missing a last activation layer before the output")]
    MissingActivationLayer,

    #[error(
        "Invalid output activation layer,
        see CostFunction::output_dependant for detailed explanation"
    )]
    WrongOutputActivationLayer,
}
