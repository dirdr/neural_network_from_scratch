use log::debug;
use ndarray::Array2;

use crate::layer::Softmax;

#[derive(Clone)]
pub enum CostFunction {
    /// The use case for CrossEntropy, is for our classification nn, taking
    /// softmax outputs and calcualting loss.
    CrossEntropy,
}

impl CostFunction {
    /// Compute the cost of the neural network
    /// # Arguments
    /// * `output` - the array (shape (j, 1)) of output of the network
    /// * `observed` - a one hotted encoded vector of observed values
    pub fn cost(&self, output: &Array2<f64>, observed: &Array2<f64>) -> f64 {
        match self {
            Self::CrossEntropy => {
                let epsilon = 1e-15;
                let clipped_output = output.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));
                let correct_class = observed.iter().position(|&x| x == 1.0).unwrap();
                -f64::ln(clipped_output[[correct_class, 0]])
            }
        }
    }

    /// Compute and return the gradient of the cost function (shape (j, 1)) with respect to `output`
    /// # Arguments
    /// * `output` - the array (shape (j, 1)) of output of the network
    /// * `observed` - a one hotted encoded vector of observed values
    pub fn cost_output_gradient(
        &self,
        output: &Array2<f64>,
        observed: &Array2<f64>,
    ) -> Array2<f64> {
        match self {
            // the gradient of the cross entropy with respect to the logit
            // is given by : dc/dz = s - y in vector notation.
            // We use this expression over the one that give dc/ds with s the softmax output
            // because the calculation is easy and that prevent us for back propagating through the
            // softmax function
            Self::CrossEntropy => Softmax::transform(output) - observed,
        }
    }
}
