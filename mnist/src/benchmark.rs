use log::{info, warn};
use ndarray::{Array2, ArrayD, Axis};
use nn_lib::neural_network::NeuralNetwork;

use crate::dataset::{load_dataset, MnistData};
use crate::network::build_neural_net;

pub struct MnistBenchmark {
    network: NeuralNetwork,
    data: MnistData,
}

impl MnistBenchmark {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            network: build_neural_net()?,
            data: load_dataset()?,
        })
    }

    fn prepare_data(&self) {
        todo!()
    }

    fn start(&self) {}

    // fn evaluate_indexed_data_point_prob(&mut self, data_index: usize) -> anyhow::Result<()> {
    //     let input = self.x.select(Axis(0), &[data_index]);
    //     let observed = self.y.select(Axis(0), &[data_index]);
    //     let output = self.network.predict(&input)?;
    //
    //     let digit = observed.iter().position(|&e| e == 1.0).unwrap();
    //
    //     info!(
    //         "prediction for digit {} : {:?}",
    //         digit,
    //         output.as_slice().unwrap()
    //     );
    //     Ok(())
    // }
    //
    // fn evaluate_indexed_data_point_short(&mut self, data_index: usize) -> anyhow::Result<()> {
    //     let input = self.x.select(Axis(0), &[data_index]);
    //     let observed = self.y.select(Axis(0), &[data_index]);
    //     let output = self.network.predict(&input)?;
    //
    //     let digit = observed.iter().position(|&e| e == 1.0).unwrap();
    //
    //     info!(
    //         "prediction for digit {} : {:?}",
    //         digit,
    //         output.as_slice().unwrap()
    //     );
    //     Ok(())
    // }
    //
    // fn evaluate_accuracy(&mut self) -> anyhow::Result<f64> {
    //     let predictions = self.network.predict(&self.x)?;
    //     let num_samples = self.x.shape()[0];
    //     let mut num_correct = 0;
    //
    //     for i in 0..num_samples {
    //         let predicted_label = predictions
    //             .slice(s![i, ..])
    //             .indexed_iter()
    //             .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    //             .map(|(idx, _)| idx);
    //
    //         let true_label = self
    //             .y
    //             .slice(s![i, ..])
    //             .indexed_iter()
    //             .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    //             .map(|(idx, _)| idx);
    //
    //         match (predicted_label, true_label) {
    //             (Some(pred), Some(true_label)) if pred == true_label => {
    //                 num_correct += 1;
    //             }
    //             (None, _) => warn!("No maximum value found in predictions for sample {}", i),
    //             (_, None) => warn!("No maximum value found in true labels for sample {}", i),
    //             _ => {}
    //         }
    //     }
    //
    //     Ok(num_correct as f64 / num_samples as f64)
    // }
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
