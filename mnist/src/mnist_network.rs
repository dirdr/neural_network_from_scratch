use nn_lib::{initialization::InitializerType, layer, neural_network::NeuralNetworkBuilder};

fn build_network() {
    let mut network_builder = NeuralNetworkBuilder::new();
    let network = network_builder
        .push_layer(Box::new(layer::DenseLayer::new(
            28 * 28,
            256,
            InitializerType::He,
        )))
        .push_layer(Box::new(layer::DenseLayer::new(
            256,
            10,
            InitializerType::He,
        )))
        .with_learning_rate(0.1)
        .with_epochs(1000);
}
