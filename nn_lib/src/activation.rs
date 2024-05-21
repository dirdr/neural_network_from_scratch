use std::fmt::Display;

use ndarray::ArrayD;

pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
}

impl Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::ReLU => f.write_str("ReLu"),
            Activation::Tanh => f.write_str("Tanh"),
            Activation::Sigmoid => f.write_str("Sigmoid"),
            Activation::Softmax => f.write_str("Softmax"),
        }
    }
}

impl Activation {
    /// Apply the activation to a vector, return a vector (shape (i, 1))
    /// # Arguments
    /// * `input` - input vector (shape (i, 1))
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

    /// Apply the derivative to a vector, return a vector (shape (i, 1))
    /// The applied derivative is with respect to each mapped vector input values
    /// Note that softmax vector function derivative is omited because the derivative of a vector function is his
    /// jacobian matrices, and in practice we don't do back propagation through softmax output
    /// layer, the gradient of the cost function is calcualted with respect to the output logits,
    /// not softmax outputs.
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
