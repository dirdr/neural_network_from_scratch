use log::info;
use candle_core::{Device, Tensor};
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layers::{ActivationLayer, DenseLayer},
    optimizer::GradientDescent,
    sequential::{Sequential, SequentialBuilder},
};

pub fn build_neural_net() -> anyhow::Result<Sequential> {
    let net = SequentialBuilder::new()
        .push(DenseLayer::new(2, 8, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(8, 1, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Sigmoid));
    Ok(net.compile(GradientDescent::new(0.02), CostFunction::BinaryCrossEntropy)?)
}

fn get_training_data() -> anyhow::Result<(Tensor, Tensor)> {
    let device = Device::cuda_if_available(0)?;
    let x_data = vec![0f64, 0f64, 0f64, 1f64, 1f64, 0f64, 1f64, 1f64];
    let y_data = vec![0f64, 1f64, 1f64, 0f64];

    let x = Tensor::from_vec(x_data, &[4, 2], &device)?;
    let y = Tensor::from_vec(y_data, &[4, 1], &device)?;
    Ok((x, y))
}

pub fn start(mut neural_network: Sequential) -> anyhow::Result<()> {
    let (x, y) = get_training_data()?;

    let (train_hist, _) = neural_network.train(
        (&x, &y),
        None,
        2000,
        1,
    )?;

    for (i, bench) in train_hist.history.iter().enumerate() {
        info!("Error for epochs {} : {}", i, bench.loss);
    }

    let predictions = neural_network.predict(&x)?;
    let predictions_vec = predictions.to_vec2::<f64>()?;

    let x_inputs = [[0f64, 0f64], [0f64, 1f64], [1f64, 0f64], [1f64, 1f64]];
    for (i, input) in x_inputs.iter().enumerate() {
        let x1 = input[0];
        let x2 = input[1];
        info!(
            "Xor prediction: {} for input {} {}",
            predictions_vec[i][0],
            x1,
            x2
        )
    }
    Ok(())
}
