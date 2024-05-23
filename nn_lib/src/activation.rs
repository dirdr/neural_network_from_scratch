use log::error;
use ndarray::{Array1, ArrayD, ArrayView1, Axis};

fn check_nan(array: &ArrayD<f64>, operation: &str) {
    if array.iter().any(|&x| x.is_nan()) {
        error!("NaN detected after {} operation", operation);
    }
}

#[derive(Debug)]
pub enum Activation {
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
}

impl Activation {
    /// Apply the activation function to each element of a multi-dimensional array
    /// dimensions doesn't matter as the tranformation is applied element wise
    /// except for the softmax function, the softmax will be computed onto each batch independently
    /// if the array if of shape (n, i) with **n** the number of batch and **i** the size of the
    /// vector, the function will return a matrices of same shape, with softmax function computed
    /// for every element in the outer-most dimension.
    /// # Arguments
    /// * `input` - a multi-dimensional array;
    pub fn apply(&self, input: &ArrayD<f64>) -> ArrayD<f64> {
        let result = match self {
            Self::ReLU => input.mapv(|e| 0f64.max(e)),
            Self::Tanh => input.mapv(|e| e.tanh()),
            Self::Sigmoid => input.mapv(|e| 1.0 / (1.0 + f64::exp(-e))),
            Self::Softmax => {
                let mut result = input.clone();
                for mut row in result.axis_iter_mut(Axis(0)) {
                    let row_as_view1: ArrayView1<f64> = row.view().into_dimensionality().unwrap();
                    let max_logit = row_as_view1.fold(f64::NEG_INFINITY, |max, &val| max.max(val));
                    let exps: Array1<f64> =
                        row_as_view1.mapv(|x| f64::exp(x - max_logit)).to_owned();
                    let sum_exps: f64 = exps.sum() + 1e-10; // to avoid division by zero
                    let softmax_row: Array1<f64> = exps.mapv(|x| x / sum_exps);
                    row.assign(&softmax_row);
                }
                result
            }
        };
        check_nan(&result, &format!("{:?}", self));
        result
    }

    /// Apply the activation function derivative to each element of a multi-dimensional array
    /// not that the dimensions doesn't matter as the tranformation is applied element wise.
    /// # Arguments
    /// * `input` - a multi-dimensional array;
    pub fn apply_derivative(&self, input: &ArrayD<f64>) -> ArrayD<f64> {
        let result = match self {
            Self::ReLU => input.mapv(|e| if e > 0f64 { 1f64 } else { 0f64 }),
            Self::Tanh => input.mapv(|e| 1f64 - e.tanh().powi(2)),
            Self::Sigmoid => {
                let sigmoid_output = self.apply(input);
                &sigmoid_output * &(1.0 - &sigmoid_output)
            }
            Self::Softmax => unimplemented!("We don't use the softmax jacobian matrix in practice"),
        };
        check_nan(&result, &format!("{:?}", self));
        result
    }
}
