use log::info;
use ndarray::{Array2, Array3};
use nn_lib::{
    activation::Activation,
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
        // TODO gérer pour les fonctions de couts "output Dependant" ne pas prendre en compte la
        // dernière activation
        .push(ActivationLayer::from(Activation::Sigmoid))
        .build(GradientDescent::new(0.1), CostFunction::Mse)?)
}

fn get_training_data() -> (Array3<f64>, Array3<f64>) {
    let x_flat = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let y_flat = vec![0.0, 1.0, 1.0, 0.0];
    (
        Array3::from_shape_vec((4, 2, 1), x_flat).unwrap(),
        Array3::from_shape_vec((4, 1, 1), y_flat).unwrap(),
    )
}

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    let (x, y) = get_training_data();
    neural_network.train_par(x.clone(), y, 1000);

    let predictions = x
        .outer_iter()
        .map(|e| neural_network.predict(e.to_owned()))
        .collect::<Vec<Array2<f64>>>();

    for (i, x) in x.clone().outer_iter().enumerate() {
        let x1 = x[[0, 0]];
        let x2 = x[[1, 0]];
        info!(
            "Xor prediction: {} for input {} {}",
            predictions[i][[0, 0]],
            x1,
            x2
        )
    }
    Ok(())
}
