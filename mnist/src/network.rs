use log::info;
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, DenseLayer},
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
    optimizer::GradientDescent,
};
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};

use crate::benchmark::MnistBenchmark;

pub fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
    let net = NeuralNetworkBuilder::new()
        .push(DenseLayer::new(28 * 28, 64, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(64, 32, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(32, 10, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Softmax));
    Ok(net.compile(GradientDescent::new(0.01), CostFunction::CrossEntropy)?)
}

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    //     let (x_train, y_train) = get_training_data()?;
    //     let (x_test, y_test) = get_test_data()?;
    //
    //     // Train the network
    //     neural_network.train(x_train.into_dyn(), y_train.into_dyn(), 100, 10)?;
    //
    //     // TODO faire une méthode qui permet de prédire dans le réseau sans avoir besoin d'être mutable
    //     // et de sauvegarder l'input comme pour une étapge de gradient descent
    //     let mut evaluator = MnistBenchmark::new(neural_network, x_test.into_dyn(), y_test.into_dyn());
    //
    //     // Evaluate the network on the test data
    //     let accuracy = evaluator.evaluate_accuracy()?;
    //     info!("Test set accuracy: {:.2}%", accuracy * 100.0);
    //
    //     let mut rng = thread_rng();
    //     let range = 0..10000;
    //     let uniform = Uniform::from(range);
    //     let count = 10;
    //     (0..count)
    //         .map(|_| uniform.sample(&mut rng))
    //         .for_each(|index| {
    //             evaluator.evaluate_indexed_data_point_prob(index).unwrap();
    //         });
    //
    //     Ok(())
    todo!()
}
