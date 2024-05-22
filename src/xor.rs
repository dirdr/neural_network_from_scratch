use log::info;
use ndarray::{arr1, arr2, arr3, Array1, Array2, Array3, ArrayD};
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, DenseLayer, LayerError},
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
    optimizer::GradientDescent,
};

pub fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
    Ok(NeuralNetworkBuilder::new()
        .push(DenseLayer::new(2, 4, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(4, 1, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Sigmoid))
        .build(GradientDescent::new(0.05), CostFunction::BinaryCrossEntropy)?)
}

fn get_training_data() -> (Array2<f64>, Array1<f64>) {
    let x = arr2(&[[0f64, 0f64], [0f64, 1f64], [1f64, 0f64], [1f64, 1f64]]);
    let y = arr1(&[0f64, 1f64, 1f64, 0f64]);
    (x, y)
}

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    let (x, y) = get_training_data();

    neural_network.train(x.clone().into_dyn(), y.into_dyn(), 5000, 2)?;

    let predictions = x
        .outer_iter()
        .map(|x| neural_network.predict(&x.to_owned().into_dyn()))
        .collect::<Result<Vec<ArrayD<_>>, LayerError>>()?;

    for (i, x) in x.clone().outer_iter().enumerate() {
        let x1 = x[0];
        let x2 = x[1];
        info!(
            "Xor prediction: {} for input {} {}",
            predictions[i][[0, 0]],
            x1,
            x2
        )
    }
    Ok(())
}
