use log::{info, trace};
use ndarray::ArrayD;
use candle_core::{Device, Tensor};
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layers::{ActivationLayer, ConvolutionalLayer, DenseLayer, MaxPoolingLayer, ReshapeLayer},
    metrics::{Metrics, MulticlassMetricType},
    optimizer::GradientDescent,
    sequential::{Sequential, SequentialBuilder},
};

use crate::{augments::augment_dataset, dataset::load_dataset};

pub enum NetType {
    Mlp,
    Conv,
}

pub fn get_neural_net(net_type: NetType, device: &Device) -> anyhow::Result<Sequential> {
    match net_type {
        NetType::Mlp => build_mlp_net(device),
        NetType::Conv => build_conv_net(device),
    }
}

fn build_conv_net(device: &Device) -> anyhow::Result<Sequential> {
    let metrics = Metrics::multiclass_classification(&vec![MulticlassMetricType::Accuracy]);

    let net = SequentialBuilder::new()
        .push(ReshapeLayer::new(&[28 * 28], &[28, 28, 1])?)
        .push(ConvolutionalLayer::new(
            (28, 28, 1),
            (3, 3),
            5,
            InitializerType::He,
            device,
        ))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(MaxPoolingLayer::new((26, 26, 5), (2, 2)))
        .push(ReshapeLayer::new(&[13, 13, 5], &[13 * 13 * 5])?)
        .push(DenseLayer::new(
            13 * 13 * 5,
            100,
            InitializerType::GlorotUniform,
            device,
        ))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(100, 10, InitializerType::GlorotUniform, device))
        .push(ActivationLayer::from(Activation::Softmax))
        .with_metrics(metrics);
    Ok(net.compile(GradientDescent::new(0.01), CostFunction::CrossEntropy)?)
}

fn build_mlp_net(device: &Device) -> anyhow::Result<Sequential> {
    let metrics = Metrics::multiclass_classification(&vec![
        MulticlassMetricType::Accuracy,
        MulticlassMetricType::MacroRecall,
        MulticlassMetricType::MacroPrecision,
        MulticlassMetricType::MacroF1Score,
        MulticlassMetricType::TypeIError,
        MulticlassMetricType::TypeIIError,
        MulticlassMetricType::Specificity,
        MulticlassMetricType::WeightedRecall,
        MulticlassMetricType::WeightedPrecision,
        MulticlassMetricType::WeightedF1Score,
    ]);

    let net = SequentialBuilder::new()
        .push(DenseLayer::new(784, 256, InitializerType::He, device))
        .push(DenseLayer::new(256, 128, InitializerType::He, device))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(128, 10, InitializerType::He, device))
        .push(ActivationLayer::from(Activation::Softmax))
        .with_metrics(metrics);
    Ok(net.compile(GradientDescent::new(0.1), CostFunction::CrossEntropy)?)
}

#[derive(Debug, Clone)]
struct PreparedDataSet {
    train: (Tensor, Tensor),
    validation: (Tensor, Tensor),
    test: (Tensor, Tensor),
}

impl PreparedDataSet {
    pub fn get_train_ref(&self) -> (&Tensor, &Tensor) {
        (&self.train.0, &self.train.1)
    }

    pub fn get_validation_ref(&self) -> (&Tensor, &Tensor) {
        (&self.validation.0, &self.validation.1)
    }

    pub fn get_test_ref(&self) -> (&Tensor, &Tensor) {
        (&self.test.0, &self.test.1)
    }
}

fn get_data(augment: bool, device: &Device) -> anyhow::Result<PreparedDataSet> {
    let mut dataset = load_dataset()?;

    if augment {
        dataset.training.0 = augment_dataset(&dataset.training.0);
    }

    let (x_train_full, y_train_full) = prepare_data(dataset.training, device)?;
    let (x_test, y_test) = prepare_data(dataset.test, device)?;

    // Split training dataset into training / validation (48000 train, 12000 validation)
    let x_train = x_train_full.narrow(0, 0, 48000)?;
    let y_train = y_train_full.narrow(0, 0, 48000)?;

    let x_validation = x_train_full.narrow(0, 48000, 12000)?;
    let y_validation = y_train_full.narrow(0, 48000, 12000)?;

    Ok(PreparedDataSet {
        train: (x_train, y_train),
        validation: (x_validation, y_validation),
        test: (x_test, y_test),
    })
}

pub fn start(
    neural_network: &mut Sequential,
    batch_size: usize,
    epochs: usize,
    augment: bool,
    device: &Device,
) -> anyhow::Result<()> {
    let prepared = get_data(augment, device)?;

    let (train_hist, validation_hist) = neural_network.train(
        prepared.get_train_ref(),
        Some(prepared.get_validation_ref()),
        epochs,
        batch_size,
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
            .get_metric_time_series(MulticlassMetricType::Accuracy)
            .unwrap()
    );

    trace!(
        "train loss by epochs {:?}",
        train_hist.get_loss_time_series()
    );
    trace!(
        "train accuracy by epochs {:?}",
        train_hist
            .get_metric_time_series(MulticlassMetricType::Accuracy)
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

        if let Some(accuracy) = train.metrics.get_metric(MulticlassMetricType::Accuracy) {
            info!(
                "network train accuracy for epoch {} : {:.2}%",
                i,
                accuracy * 100f64
            );
        }
        if let Some(accuracy) = validation.metrics.get_metric(MulticlassMetricType::Accuracy) {
            info!(
                "network validation accuracy for epoch {} : {:.2}%",
                i,
                accuracy * 100f64
            );
        }

        if let Some(macro_recall) = train.metrics.get_metric(MulticlassMetricType::MacroRecall) {
            info!("network train macro recall for epoch {} : {:.4}", i, macro_recall);
        }
        if let Some(macro_recall) = validation.metrics.get_metric(MulticlassMetricType::MacroRecall) {
            info!("network validation macro recall for epoch {} : {:.4}", i, macro_recall);
        }

        if let Some(macro_precision) = train.metrics.get_metric(MulticlassMetricType::MacroPrecision) {
            info!("network train macro precision for epoch {} : {:.4}", i, macro_precision);
        }
        if let Some(macro_precision) = validation.metrics.get_metric(MulticlassMetricType::MacroPrecision) {
            info!("network validation macro precision for epoch {} : {:.4}", i, macro_precision);
        }

        if let Some(macro_f1) = train.metrics.get_metric(MulticlassMetricType::MacroF1Score) {
            info!("network train macro F1 score for epoch {} : {:.4}", i, macro_f1);
        }
        if let Some(macro_f1) = validation.metrics.get_metric(MulticlassMetricType::MacroF1Score) {
            info!("network validation macro F1 score for epoch {} : {:.4}", i, macro_f1);
        }

        if let Some(type1_error) = train.metrics.get_metric(MulticlassMetricType::TypeIError) {
            info!("network train Type I error for epoch {} : {:.4}", i, type1_error);
        }
        if let Some(type1_error) = validation.metrics.get_metric(MulticlassMetricType::TypeIError) {
            info!("network validation Type I error for epoch {} : {:.4}", i, type1_error);
        }

        if let Some(type2_error) = train.metrics.get_metric(MulticlassMetricType::TypeIIError) {
            info!("network train Type II error for epoch {} : {:.4}", i, type2_error);
        }
        if let Some(type2_error) = validation.metrics.get_metric(MulticlassMetricType::TypeIIError) {
            info!("network validation Type II error for epoch {} : {:.4}", i, type2_error);
        }

        if let Some(specificity) = train.metrics.get_metric(MulticlassMetricType::Specificity) {
            info!("network train specificity for epoch {} : {:.4}", i, specificity);
        }
        if let Some(specificity) = validation.metrics.get_metric(MulticlassMetricType::Specificity) {
            info!("network validation specificity for epoch {} : {:.4}", i, specificity);
        }

        if let Some(weighted_recall) = train.metrics.get_metric(MulticlassMetricType::WeightedRecall) {
            info!("network train weighted recall for epoch {} : {:.4}", i, weighted_recall);
        }
        if let Some(weighted_recall) = validation.metrics.get_metric(MulticlassMetricType::WeightedRecall) {
            info!("network validation weighted recall for epoch {} : {:.4}", i, weighted_recall);
        }

        if let Some(weighted_precision) = train.metrics.get_metric(MulticlassMetricType::WeightedPrecision) {
            info!("network train weighted precision for epoch {} : {:.4}", i, weighted_precision);
        }
        if let Some(weighted_precision) = validation.metrics.get_metric(MulticlassMetricType::WeightedPrecision) {
            info!("network validation weighted precision for epoch {} : {:.4}", i, weighted_precision);
        }

        if let Some(weighted_f1) = train.metrics.get_metric(MulticlassMetricType::WeightedF1Score) {
            info!("network train weighted F1 score for epoch {} : {:.4}", i, weighted_f1);
        }
        if let Some(weighted_f1) = validation.metrics.get_metric(MulticlassMetricType::WeightedF1Score) {
            info!("network validation weighted F1 score for epoch {} : {:.4}", i, weighted_f1);
        }

        info!("\n");
    }

    let bench = neural_network.evaluate(prepared.get_test_ref(), 10);

    info!("loss for test data : {}", bench.loss);

    if let Some(accuracy) = bench.metrics.get_metric(MulticlassMetricType::Accuracy) {
        info!("network test accuracy : {:.2}%", accuracy * 100f64);
    }

    if let Some(macro_recall) = bench.metrics.get_metric(MulticlassMetricType::MacroRecall) {
        info!("network test macro recall : {:.4}", macro_recall);
    }

    if let Some(macro_precision) = bench.metrics.get_metric(MulticlassMetricType::MacroPrecision) {
        info!("network test macro precision : {:.4}", macro_precision);
    }

    if let Some(macro_f1) = bench.metrics.get_metric(MulticlassMetricType::MacroF1Score) {
        info!("network test macro F1 score : {:.4}", macro_f1);
    }

    if let Some(type1_error) = bench.metrics.get_metric(MulticlassMetricType::TypeIError) {
        info!("network test Type I error : {:.4}", type1_error);
    }

    if let Some(type2_error) = bench.metrics.get_metric(MulticlassMetricType::TypeIIError) {
        info!("network test Type II error : {:.4}", type2_error);
    }

    if let Some(specificity) = bench.metrics.get_metric(MulticlassMetricType::Specificity) {
        info!("network test specificity : {:.4}", specificity);
    }

    if let Some(weighted_recall) = bench.metrics.get_metric(MulticlassMetricType::WeightedRecall) {
        info!("network test weighted recall : {:.4}", weighted_recall);
    }

    if let Some(weighted_precision) = bench.metrics.get_metric(MulticlassMetricType::WeightedPrecision) {
        info!("network test weighted precision : {:.4}", weighted_precision);
    }

    if let Some(weighted_f1) = bench.metrics.get_metric(MulticlassMetricType::WeightedF1Score) {
        info!("network test weighted F1 score : {:.4}", weighted_f1);
    }

    Ok(())
}

fn prepare_data(data: (ArrayD<u8>, ArrayD<u8>), device: &Device) -> anyhow::Result<(Tensor, Tensor)> {
    // Normalize images: convert u8 to f64 and divide by 255
    let x = data.0.mapv(|e| e as f64 / 255.0);
    let num_samples = x.shape()[0];

    // Reshape to (num_samples, 784)
    let x_flat: Vec<f64> = x.into_iter().collect();
    let x_tensor = Tensor::from_vec(x_flat, &[num_samples, 28 * 28], device)?;

    // One-hot encode labels
    let y_tensor = one_hot_encode(&data.1, 10, device)?;

    Ok((x_tensor, y_tensor))
}

fn one_hot_encode(labels: &ArrayD<u8>, num_classes: usize, device: &Device) -> anyhow::Result<Tensor> {
    let num_labels = labels.len();
    let mut one_hot_vec = vec![0.0f64; num_labels * num_classes];

    for (i, &label) in labels.iter().enumerate() {
        one_hot_vec[i * num_classes + label as usize] = 1.0;
    }

    let tensor = Tensor::from_vec(one_hot_vec, &[num_labels, num_classes], device)?;
    Ok(tensor)
}
