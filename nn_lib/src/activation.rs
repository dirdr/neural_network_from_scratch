use ndarray::ArrayD;

pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
}

impl Activation {
    /// Apply the activation function to each element of a multi-dimensional array
    /// not that the dimensions doesn't matter as the tranformation is applied element wise
    /// however certain function work naturally with specific dimension like `Activation::Softmax`
    /// # Arguments
    /// * `input` - a multi-dimensional array;
    pub fn apply(&self, input: &ArrayD<f64>) -> ArrayD<f64> {
        match self {
            Self::ReLU => input.mapv(|e| 0f64.max(e)),
            Self::Tanh => input.mapv(|e| e.tanh()),
            Self::Sigmoid => input.mapv(|e| 1.0 / (1.0 + f64::exp(-e))),
            Self::Softmax => {
                let max_logit = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum_exps = input.mapv(|e| f64::exp(e - max_logit)).sum();
                input.mapv(|e| f64::exp(e - max_logit) / sum_exps)
            }
        }
    }

    /// Apply the activation function derivative to each element of a multi-dimensional array
    /// not that the dimensions doesn't matter as the tranformation is applied element wise.
    /// # Arguments
    /// * `input` - a multi-dimensional array;
    pub fn apply_derivative(&self, input: &ArrayD<f64>) -> ArrayD<f64> {
        match self {
            Self::ReLU => input.mapv(|e| if e > 0f64 { 1f64 } else { 0f64 }),
            Self::Tanh => input.mapv(|e| 1f64 - e.tanh().powi(2)),
            Self::Sigmoid => {
                let sigmoid_output = self.apply(input);
                &sigmoid_output * &(1.0 - &sigmoid_output)
            }
            Self::Softmax => unimplemented!("We don't use the softmax jacobian matrix in practice"),
        }
    }
}
