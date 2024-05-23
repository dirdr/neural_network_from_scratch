use log::info;
use ndarray::{arr1, arr2, Array1, Array2, Axis};
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, DenseLayer},
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
    optimizer::GradientDescent,
};

pub fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
    let net = NeuralNetworkBuilder::new()
        .push(DenseLayer::new(2, 8, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(8, 1, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Sigmoid));
    Ok(net.compile(GradientDescent::new(0.02), CostFunction::BinaryCrossEntropy)?)
}

fn get_training_data() -> (Array2<f64>, Array1<f64>) {
    let x = arr2(&[[0f64, 0f64], [0f64, 1f64], [1f64, 0f64], [1f64, 1f64]]);
    let y = arr1(&[0f64, 1f64, 1f64, 0f64]);
    (x, y)
}

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    let (x, y) = get_training_data();

    let history = neural_network.train(
        x.clone().into_dyn(),
        y.insert_axis(Axis(1)).into_dyn(),
        2000,
        1,
    )?;

    for (i, bench) in history.history.iter().enumerate() {
        info!("Error for epochs {} : {}", i, bench.loss);
    }

    let predictions = neural_network.predict(&x.clone().into_dyn())?;

    for (i, x) in x.clone().outer_iter().enumerate() {
        let x1 = x[0];
        let x2 = x[1];
        info!(
            "Xor prediction: {} for input {} {}",
            predictions[[i, 0]],
            x1,
            x2
        )
    }
    Ok(())
}
