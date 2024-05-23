use log::info;
use ndarray::{Array, Array2, ArrayD};
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

use crate::dataset::{load_dataset, MnistData};

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
    let dataset = load_dataset()?;
    let (x_train, y_train) = prepare_data(dataset.training)?;
    let (x_test, y_test) = prepare_data(dataset.test)?;

    let history = neural_network.train(x_train.into_dyn(), y_train.into_dyn(), 100, 10)?;

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

fn prepare_data(data: (ArrayD<u8>, ArrayD<u8>)) -> anyhow::Result<(Array2<f64>, Array2<f64>)> {
    let x = data.0.mapv(|e| e as f64 / 255f64);
    let outer = x.shape()[0];
    let x = x.into_shape((outer, 28 * 28))?;
    let y = one_hot_encode(&data.1, 10);
    Ok((x, y))
}

fn one_hot_encode(labels: &ArrayD<u8>, num_classes: usize) -> Array2<f64> {
    let num_labels = labels.len();
    let mut one_hot = Array2::<f64>::zeros((num_labels, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        one_hot[[i, label as usize]] = 1.0;
    }
    one_hot
}
