use nn_lib::{
    initialization::InitializerType,
    layer::{ActivationLayer, ActivationType, DenseLayer, Layer},
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
};

/// This simple xor neural netowork is used to :
/// 1. test our neural network with predictable output
/// 2. improve performances
struct Xor;

impl Xor {
    fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
        Ok(NeuralNetworkBuilder::new()
            .push_layer(2, 1, InitializerType::He)
            .push_layer(ActivationLayer::from(ActivationType::ReLU))
            .with_cost_function(cost_function)
            .with_learning_rate(0.1)
            .with_epochs(0.1)
            .build()?)
    }
}

fn xor_test_neural_network() -> anyhow::Result<()> {
    let mut xor_network_buildr = NeuralNetworkBuilder::new();
    xor_network_buildr
        .push_layer(DenseLayer::new(2, 1, InitializerType::He))
        .buld()?;
    Ok(())
}
