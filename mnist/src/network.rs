use log::{debug, info, trace};
use ndarray::{s, Array2, ArrayD};
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, ConvolutionalLayer, DenseLayer, ReshapeLayer},
    metrics::MetricsType,
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
    optimizer::GradientDescent,
};

use crate::dataset::load_dataset;

pub enum NetType {
    Mlp,
    Conv,
}

pub fn get_neural_net(net_type: NetType) -> anyhow::Result<NeuralNetwork> {
    match net_type {
        NetType::Mlp => build_mlp_net(),
        NetType::Conv => build_conv_net(),
    }
}

fn build_conv_net() -> anyhow::Result<NeuralNetwork> {
    let net = NeuralNetworkBuilder::new()
        .watch(MetricsType::Accuracy)
        .push(ReshapeLayer::new(&[28 * 28], &[28, 28, 1])?)
        .push(ConvolutionalLayer::new(
            (28, 28, 1),
            (3, 3),
            5,
            InitializerType::He,
        ))
        .push(ActivationLayer::from(Activation::Sigmoid))
        .push(ReshapeLayer::new(&[26, 26, 5], &[26 * 26 * 5])?)
        .push(DenseLayer::new(
            26 * 26 * 5,
            100,
            InitializerType::GlorotUniform,
        ))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(100, 10, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Softmax));
    Ok(net.compile(GradientDescent::new(0.01), CostFunction::CrossEntropy)?)
}

fn build_mlp_net() -> anyhow::Result<NeuralNetwork> {
    let net = NeuralNetworkBuilder::new()
        .watch(MetricsType::Accuracy)
        .push(DenseLayer::new(
            28 * 28,
            256,
            InitializerType::GlorotUniform,
        ))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(256, 10, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Softmax));
    Ok(net.compile(GradientDescent::new(0.01), CostFunction::CrossEntropy)?)
}

pub fn start(neural_network: &mut NeuralNetwork) -> anyhow::Result<()> {
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
        10,
        128,
    )?;

    trace!(
        "validation loss by epochs {:?}",
        validation_hist.as_ref().unwrap().get_loss_time_series()
    );
    trace!(
        "validation accuracy by epochs {:?}",
        validation_hist
            .as_ref()
            .unwrap()
            .get_metric_time_series(MetricsType::Accuracy)
            .unwrap()
    );

    trace!(
        "train loss by epochs {:?}",
        train_hist.get_loss_time_series()
    );
    trace!(
        "train accuracy by epochs {:?}",
        train_hist
            .get_metric_time_series(MetricsType::Accuracy)
            .unwrap()
    );

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
        info!("\n");
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
