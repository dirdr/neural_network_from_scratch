use mnist::build_network;
use nn_lib::{
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, ActivationType, DenseLayer, Layer},
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
};

/// This simple xor neural netowork is used to :
/// 1. test our neural network with predictable output
/// 2. improve performances
struct Xor {
    neural_network: NeuralNetwork,
}

impl Xor {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {
            neural_network: Self::build_neural_net()?,
        })
    }

    fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
        Ok(NeuralNetworkBuilder::new()
            .push_layer(DenseLayer::new(2, 1, InitializerType::He))
            .push_layer(ActivationLayer::from(ActivationType::ReLU))
            .with_cost_function(CostFunction::Mse)
            .with_learning_rate(0.1)
            .with_epochs(100)
            .build()?)
    }

    fn start(&mut self) -> anyhow::Result<()> {
        Array3::self.neural_network.train();
        Ok(())
    }
}
