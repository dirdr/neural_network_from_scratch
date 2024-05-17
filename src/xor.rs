use std::fmt;

use log::info;
use ndarray::{Array2, Array3};
use nn_lib::{
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, ActivationType, DenseLayer},
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
};

pub fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
    Ok(NeuralNetworkBuilder::new()
        .push_layer(DenseLayer::new(2, 8, InitializerType::GlorotUniform))
        .push_layer(ActivationLayer::from(ActivationType::Tanh))
        .push_layer(DenseLayer::new(8, 1, InitializerType::GlorotUniform))
        // cost function BSE assume sigmoid so don't include it after the output layer
        .with_cost_function(CostFunction::BinaryCrossEntropy)
        .with_learning_rate(0.1)
        .with_epochs(10)
        .build()?)
}

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    let x_flat = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let x = Array3::from_shape_vec((4, 2, 1), x_flat.clone())?;
    let y_flat = vec![0.0, 1.0, 1.0, 0.0];
    let y = Array3::from_shape_vec((4, 1, 1), y_flat)?;
    neural_network.train(x.clone(), y);
    // test the neural network with exemples:
    let predictions = x
        .outer_iter()
        .map(|e| neural_network.predict(e.to_owned()))
        .collect::<Vec<Array2<f64>>>();
    for i in 0..4 {
        info!(
            "Xor prediction : {} for input : {}, {}",
            predictions[i][[0, 0]],
            x_flat[i],
            x_flat[i + 1]
        );
    }
    Ok(())
}
