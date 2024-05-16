use core::panic;

use nn_lib::{
    initialization::InitializerType,
    layer::{self, ActivationType},
    neural_network::NeuralNetworkBuilder,
};

use crate::load_dataset;

fn build_network() {
    let data_set =
        load_dataset().unwrap_or_else(|_| panic!("error occured when loading the dataset"));

    let network = NeuralNetworkBuilder::new()
        .push_layer(Box::new(layer::DenseLayer::new(
            28 * 28,
            256,
            InitializerType::He,
        )))
        .push_layer(Box::new(layer::ActivationLayer::from(ActivationType::ReLU)))
        .push_layer(Box::new(layer::DenseLayer::new(
            256,
            10,
            InitializerType::He,
        )))
        .with_learning_rate(0.1)
        .with_epochs(1000)
        .build();

    if let Ok(network) = network {
        network.train()
    }
}
