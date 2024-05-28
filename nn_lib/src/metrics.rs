use std::collections::HashMap;

use ndarray::{ArrayD, Axis};
use ndarray_stats::QuantileExt;

pub struct History {
    pub history: Vec<Benchmark>,
}

impl History {
    pub fn new() -> Self {
        Self { history: vec![] }
    }

    pub fn get_loss_time_series(&self) -> Vec<f64> {
        self.history.iter().map(|h| h.loss).collect::<Vec<_>>()
    }

    pub fn get_metric_time_series(&self, metrics_type: MetricsType) -> Option<Vec<f64>> {
        self.history
            .iter()
            .map(|h| h.metrics.get_metric(metrics_type))
            .collect::<Option<Vec<_>>>()
    }
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Benchmark {
    pub metrics: Metrics,
    pub loss: f64,
}

impl Benchmark {
    pub fn new(metrics: &Vec<MetricsType>) -> Self {
        Self {
            metrics: Metrics::from(metrics),
            loss: 0f64,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum MetricsType {
    Accuracy,
    Recall,
    Precision,
}

pub struct Metrics {
    pub metrics: HashMap<MetricsType, f64>,
}

impl Metrics {
    fn from(metrics: &Vec<MetricsType>) -> Self {
        let mut map = HashMap::new();
        for el in metrics {
            map.insert(*el, 0f64);
        }
        Self { metrics: map }
    }

    pub fn get_all(&self) -> &HashMap<MetricsType, f64> {
        &self.metrics
    }

    pub fn get_metric(&self, metric: MetricsType) -> Option<f64> {
        if let Some(metric) = self.metrics.get(&metric) {
            return Some(*metric);
        }
        None
    }

    /// Accumulate metrics for a given batch
    /// # Arguments
    /// * `predictions` a batched probability distribution of shape (n, i)
    /// * `true_labels` a batched observed values of shape (n, i)
    pub fn accumulate(&mut self, predictions: &ArrayD<f64>, observed: &ArrayD<f64>) {
        for (metric_type, value) in self.metrics.iter_mut() {
            match metric_type {
                MetricsType::Accuracy => {
                    let pred_classes = predictions.map_axis(Axis(1), |prob| prob.argmax().unwrap());

                    let true_classes =
                        observed.map_axis(Axis(1), |one_hot| one_hot.argmax().unwrap());

                    let correct_preds = pred_classes
                        .iter()
                        .zip(true_classes.iter())
                        .filter(|&(pred, true_label)| pred == true_label)
                        .count();

                    let accuracy = correct_preds as f64 / predictions.shape()[0] as f64;
                    *value += accuracy;
                }
                MetricsType::Recall => {
                    todo!()
                }
                MetricsType::Precision => {
                    todo!()
                }
            }
        }
    }

    pub fn mean(&mut self, metric_type: MetricsType, number_of_batch: usize) {
        if let Some(m) = self.metrics.get_mut(&metric_type) {
            *m /= number_of_batch as f64;
        }
    }

    pub fn mean_all(&mut self, number_of_batch: usize) {
        for (&_, value) in self.metrics.iter_mut() {
            *value /= number_of_batch as f64;
        }
    }
}
