use ndarray::{Array2, Array3, ArrayD, Axis};
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, DenseLayer},
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
    optimizer::GradientDescent,
};

use crate::load_dataset;

pub fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
    let network = NeuralNetworkBuilder::new()
        .push(DenseLayer::new(28 * 28, 40, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(40, 10, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Softmax))
        .build(GradientDescent::new(0.1), CostFunction::CrossEntropy)?;
    Ok(network)
}

fn get_taining_data() -> anyhow::Result<(Array2<f64>, Array3<f64>)> {
    let data_set = load_dataset()?;
    let x = data_set.training.0.mapv(|e| e as f64 / 255f64);
    let outer = x.shape()[0];
    let x = x.into_shape((outer, 28 * 28))?;
    let y = one_hot_encode(&data_set.training.1, 10);
    // TODO pas super claire dans le workflow general des dimensions
    // TODO gérer la conversion matricielle des vecteur en matrice pour le .dot dans la lib au lieu
    // de laisser l'utilisateur le faire
    // to convert output vector from (60000, 10) to (60000, 10, 1)
    let y = y.insert_axis(Axis(2));
    Ok((x, y))
}

fn get_test_data() -> anyhow::Result<(Array2<f64>, Array3<f64>)> {
    let data_set = load_dataset()?;
    let x = data_set.test.0.mapv(|e| e as f64 / 255f64);
    let outer = x.shape()[0];
    let x = x.into_shape((outer, 28 * 28))?;
    todo!()
}

fn one_hot_encode(labels: &ArrayD<u8>, num_classes: usize) -> Array2<f64> {
    let num_labels = labels.len();
    let mut one_hot = Array2::<f64>::zeros((num_labels, num_classes));

    for (i, &label) in labels.iter().enumerate() {
        one_hot[[i, label as usize]] = 1.0;
    }

    one_hot
}

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    let (x_train, y_train) = get_taining_data()?;

    neural_network.train(x_train.into_dyn(), y_train.into_dyn(), 10, 10)?;

    Ok(())
}
