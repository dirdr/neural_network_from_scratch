use log::info;
use ndarray::{arr2, Array2, ArrayD};
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
    let x = x
        .iter()
        .cloned()
        .map(|a| a.into_dyn())
        .collect::<Vec<ArrayD<_>>>();
    let y = y
        .iter()
        .cloned()
        .map(|a| a.into_dyn())
        .collect::<Vec<ArrayD<_>>>();

    neural_network.train_par(x.clone(), y, 5000, 1)?;

    let predictions = x
        .iter()
        .cloned()
        .map(|x| neural_network.predict(&x.into_dyn()))
        .collect::<Result<Vec<ArrayD<_>>, LayerError>>()?;

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
