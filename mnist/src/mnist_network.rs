use log::{debug, info, warn};
use ndarray::{s, Array2, Array3, ArrayD, Axis};
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, DenseLayer},
    neural_network::{NeuralNetwork, NeuralNetworkBuilder},
    optimizer::GradientDescent,
};
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};

struct MnistEvaluator {
    network: NeuralNetwork,
    x: ArrayD<f64>,
    y: ArrayD<f64>,
}

impl MnistEvaluator {
    fn new(network: NeuralNetwork, x: ArrayD<f64>, y: ArrayD<f64>) -> Self {
        Self { network, x, y }
    }

    fn evaluate_indexed_data_point(&mut self, data_index: usize) -> anyhow::Result<()> {
        let input = self.x.select(Axis(0), &[data_index]);
        let observed = self.y.select(Axis(0), &[data_index]);
        let output = self.network.predict(&input)?;

        let digit = observed.iter().position(|&e| e == 1.0).unwrap();

        info!(
            "prediction for digit {} : {:?}",
            digit,
            output.as_slice().unwrap()
        );
        Ok(())
    }

    fn evaluate_accuracy(&mut self) -> anyhow::Result<f64> {
        let predictions = self.network.predict(&self.x)?;
        let num_samples = self.x.shape()[0];
        let mut num_correct = 0;

        for i in 0..num_samples {
            let predicted_label = predictions
                .slice(s![i, ..])
                .indexed_iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx);

            let true_label = self
                .y
                .slice(s![i, ..])
                .indexed_iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx);

            match (predicted_label, true_label) {
                (Some(pred), Some(true_label)) if pred == true_label => {
                    num_correct += 1;
                }
                (None, _) => warn!("No maximum value found in predictions for sample {}", i),
                (_, None) => warn!("No maximum value found in true labels for sample {}", i),
                _ => {}
            }
        }

        Ok(num_correct as f64 / num_samples as f64)
    }
}

use crate::load_dataset;

pub fn build_neural_net() -> anyhow::Result<NeuralNetwork> {
    let network = NeuralNetworkBuilder::new()
        .push(DenseLayer::new(28 * 28, 64, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(64, 32, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(32, 10, InitializerType::GlorotUniform))
        .push(ActivationLayer::from(Activation::Softmax))
        .build(GradientDescent::new(0.01), CostFunction::CrossEntropy)?;
    Ok(network)
}

fn get_training_data() -> anyhow::Result<(Array2<f64>, Array2<f64>)> {
    let data_set = load_dataset()?;
    let x = data_set.training.0.mapv(|e| e as f64 / 255f64);
    let outer = x.shape()[0];
    let x = x.into_shape((outer, 28 * 28))?;
    let y = one_hot_encode(&data_set.training.1, 10);
    Ok((x, y))
}

fn get_test_data() -> anyhow::Result<(Array2<f64>, Array2<f64>)> {
    let data_set = load_dataset()?;
    let x = data_set.test.0.mapv(|e| e as f64 / 255f64);
    let outer = x.shape()[0];
    let x = x.into_shape((outer, 28 * 28))?;
    let y = one_hot_encode(&data_set.test.1, 10);
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

pub fn start(mut neural_network: NeuralNetwork) -> anyhow::Result<()> {
    let (x_train, y_train) = get_training_data()?;
    let (x_test, y_test) = get_test_data()?;

    // Train the network
    neural_network.train(x_train.into_dyn(), y_train.into_dyn(), 100, 10)?;

    // TODO faire une méthode qui permet de prédire dans le réseau sans avoir besoin d'être mutable
    // et de sauvegarder l'input comme pour une étapge de gradient descent
    let mut evaluator = MnistEvaluator::new(neural_network, x_test.into_dyn(), y_test.into_dyn());

    // Evaluate the network on the test data
    let accuracy = evaluator.evaluate_accuracy()?;
    info!("Test set accuracy: {:.2}%", accuracy * 100.0);

    let mut rng = thread_rng();
    let range = 0..10000;
    let uniform = Uniform::from(range);
    let count = 10;
    (0..count)
        .map(|_| uniform.sample(&mut rng))
        .for_each(|index| {
            evaluator.evaluate_indexed_data_point(index).unwrap();
        });

    Ok(())
}
