use std::fmt;

use log::{debug, info};
use ndarray::{Array2, Array3};
use nn_lib::{
    activations::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, DenseLayer},
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
};

pub fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
    Ok(NeuralNetworkBuilder::new()
        .push_layer(DenseLayer::new(2, 16, InitializerType::GlorotUniform))
        .push_layer(ActivationLayer::from(Activation::ReLU))
        .push_layer(DenseLayer::new(16, 1, InitializerType::GlorotUniform))
        // cost function BSE assume sigmoid so don't include it after the output layer
        .push_layer(ActivationLayer::from(Activation::Sigmoid))
        .with_cost_function(CostFunction::Mse)
        .with_learning_rate(0.1)
        .with_epochs(1000)
        .build()?)
}

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    let x_flat = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let x = Array3::from_shape_vec((4, 2, 1), x_flat.clone())?;
    let y_flat = vec![0.0, 1.0, 1.0, 0.0];
    let y = Array3::from_shape_vec((4, 1, 1), y_flat)?;
    neural_network.train_par(x.clone(), y);
    // test the neural network with exemples:
    let predictions = x
        .outer_iter()
        .map(|e| neural_network.predict(e.to_owned()))
        .collect::<Vec<Array2<f64>>>();

    debug!("x input {:?}", x);
    // a vérifier que c'est bon cette histoire
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
