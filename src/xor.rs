use std::fmt;

use log::{debug, info};
use ndarray::{Array2, Array3};
use nn_lib::{
    activations::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, DenseLayer},
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
    optimizer::GradientDescent,
};

pub fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
    Ok(NeuralNetworkBuilder::new()
        .push(DenseLayer::new(2, 16, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(16, 1, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Sigmoid))
        .build(GradientDescent::new(0.1), CostFunction::BinaryCrossEntropy))
}

fn get_training_data() -> (Array3<f64>, Array3<f64>) {
    let x_flat = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y_flat = vec![0.0, 1.0, 1.0, 0.0];
    (
        Array3::from_shape_vec((4, 2, 1), x_flat.clone())?,
        Array3::from_shape_vec((4, 1, 1), y_flat)?,
    )
}

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    let (x, y) = get_training_data();
    neural_network.train_par(x.clone(), y, 1000);

    let predictions = x
        .outer_iter()
        .map(|e| neural_network.predict(e.to_owned()))
        .collect::<Vec<Array2<f64>>>();

    for (i, chunk) in x_flat.chunks_exact(2).enumerate() {
        let x1 = chunk[0];
        let x2 = chunk[1];
        info!(
            "Xor prediction: {} for input: {}, {}",
            predictions[i][[0, 0]],
            x1,
            x2
        );
    }
    Ok(())
}
