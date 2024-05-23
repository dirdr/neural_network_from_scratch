use log::{debug, info};
use ndarray::{s, Array2, ArrayD};
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, DenseLayer},
    metrics::MetricsType,
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
    optimizer::GradientDescent,
};

use crate::dataset::load_dataset;

pub fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
    let net = NeuralNetworkBuilder::new()
        .push(DenseLayer::new(
            28 * 28,
            512,
            InitializerType::GlorotUniform,
        ))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(512, 256, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(256, 128, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(128, 64, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(64, 32, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(32, 10, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Softmax))
        .watch(MetricsType::Accuracy);
    Ok(net.compile(GradientDescent::new(0.01), CostFunction::CrossEntropy)?)
}

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    let dataset = load_dataset()?;
    let (x_train, y_train) = prepare_data(dataset.training)?;

    // split the training dataset into training / validation
    let (x_test, y_test) = prepare_data(dataset.test)?;

    let (x_validation, y_validation) = (
        x_train.slice(s![48000..60000, ..]),
        y_train.slice(s![48000..60000, ..]),
    );
    let (x_train, y_train) = (
        x_train.slice(s![0..48000, ..]),
        y_train.slice(s![0..48000, ..]),
    );

    let (train_hist, validation_hist) = neural_network.train(
        (
            &x_train.to_owned().into_dyn(),
            &y_train.to_owned().into_dyn(),
        ),
        Some((
            &x_validation.to_owned().into_dyn(),
            &y_validation.to_owned().into_dyn(),
        )),
        20,
        128,
    )?;

    for (i, (train, validation)) in train_hist
        .history
        .iter()
        .zip(validation_hist.unwrap().history.iter())
        .enumerate()
    {
        info!("train loss for epochs {} : {}", i, train.loss);
        info!("validation loss for epochs {} : {}", i, validation.loss);
        if let Some(accuracy) = train.metrics.get_metric(MetricsType::Accuracy) {
            info!(
                "network train accuracy for epoch {} : {:.2}%",
                i,
                accuracy * 100f64
            );
        } else {
            debug!("accuracy has not been set")
        }
        if let Some(accuracy) = validation.metrics.get_metric(MetricsType::Accuracy) {
            info!(
                "network validation accuracy for epoch {} : {:.2}%",
                i,
                accuracy * 100f64
            );
        } else {
            debug!("accuracy has not been set")
        }
    }

    // evaluate model on test data
    let bench = neural_network.evaluate(&x_test.into_dyn(), &y_test.into_dyn(), 10);

    info!("loss for test data : {}", bench.loss);
    if let Some(accuracy) = bench.metrics.get_metric(MetricsType::Accuracy) {
        info!("network accuracy : {}", accuracy);
    } else {
        debug!("accuracy has not been set")
    }

    Ok(())
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
