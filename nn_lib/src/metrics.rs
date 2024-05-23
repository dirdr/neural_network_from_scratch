use std::collections::HashMap;

use ndarray::ArrayD;

pub struct History {
    pub history: Vec<Benchmark>,
}

impl History {
    pub fn new() -> Self {
        Self { history: vec![] }
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

    pub fn update(&mut self, predictions: &ArrayD<f64>, true_labels: &ArrayD<f64>) {
        for (metric_type, value) in self.metrics.iter_mut() {
            match metric_type {
                MetricsType::Accuracy => {
                    todo!()
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
}
