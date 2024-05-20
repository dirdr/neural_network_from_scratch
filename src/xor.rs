use log::info;
use ndarray::{arr2, Array2};
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
        .push(DenseLayer::new(2, 8, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(8, 1, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Sigmoid))
        .build(GradientDescent::new(0.05), CostFunction::BinaryCrossEntropy)?)
}

fn get_training_data() -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
    let x_data = vec![
        arr2(&[[0.0], [0.0]]),
        arr2(&[[0.0], [1.0]]),
        arr2(&[[1.0], [0.0]]),
        arr2(&[[1.0], [1.0]]),
    ];

    let y_data = vec![
        arr2(&[[0.0]]),
        arr2(&[[1.0]]),
        arr2(&[[1.0]]),
        arr2(&[[0.0]]),
    ];
    (x_data, y_data)
}

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    let (x, y) = get_training_data();
    neural_network.train_par(x.clone(), y, 5000, 1);

    let predictions = x
        .iter()
        .map(|x| neural_network.predict(x))
        .collect::<Vec<Array2<f64>>>();

    for (i, x) in x.clone().iter().enumerate() {
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
