use ndarray::Array2;

use crate::activation::Activation;

#[derive(Copy, Clone)]
pub enum CostFunction {
    /// The use case for CrossEntropy, is for our classification nn, taking
    /// softmax outputs and calculating loss.
    CrossEntropy,
    BinaryCrossEntropy,
    Mse,
}

impl CostFunction {
    /// This crate don't use any kind of auto diff mechanism,
    /// thus, for function like BinaryCrossEntropy and CrossEntropy that need clamped output,
    /// we assume Sigmoid and Softmax respectively as the output activation layer.
    /// the gradient calculation is done with those activation function in mind.
    /// Those function are called 'Output dependant' to constrast with function like Mse, of which
    /// the derivative can be easily calculated with respect to any output layer, because it
    /// doesn't need clamped output.
    pub fn output_dependant(&self) -> bool {
        match self {
            Self::BinaryCrossEntropy | Self::CrossEntropy => true,
            Self::Mse => false,
        }
    }

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
            Self::BinaryCrossEntropy => {
                let epsilon = 1e-15;
                let clipped_output = output.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));
                let log_loss = observed * &clipped_output.mapv(f64::ln)
                    + (1.0 - observed) * &((1.0 - clipped_output).mapv(f64::ln));
                -log_loss.mean().unwrap()
            }
            Self::Mse => {
                let diff = output - observed;
                diff.mapv(|x| x.powi(2)).mean().unwrap()
            }
        }
    }

    /// Compute and return the gradient of the cost function (shape (j, 1)) with respect to `output`
    /// Note that this simple, from scratch library, don't use auto-differentiation
    /// so for : `BinaryCrossEntropy` the calculation use a Sigmoid activation function as the last
    /// layer, and for `CrossEntropy` the calculation use a Softmax activation function as the last
    /// layer
    /// # Arguments
    /// * `output` - the array (shape (j, 1)) of output of the network
    /// * `observed` - a one hotted encoded vector of observed values
    pub fn cost_output_gradient(
        &self,
        output: &Array2<f64>,
        observed: &Array2<f64>,
    ) -> Array2<f64> {
        match self {
            // the gradient of the cross entropy with respect to the logits
            // is given by : dc/dz = s - y in vector notation.
            // We use this expression over the one that give dc/ds with s the softmax output
            // because the calculation is easy and that prevent us for back propagating through the
            // softmax function
            Self::CrossEntropy => {
                let softmax = Activation::Softmax;
                softmax.apply(output) - observed
            }
            Self::BinaryCrossEntropy => {
                let sigmoid = Activation::Sigmoid;
                sigmoid.apply(output) - observed
            }
            Self::Mse => 2f64 * (output - observed) / output.len() as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cross_entropy() {
        let output = array![[0.7], [0.2], [0.1]];
        let observed = array![[0.0], [1.0], [0.0]];
        let cost_function = CostFunction::CrossEntropy;
        let cost = cost_function.cost(&output, &observed);
        assert!(cost.is_finite());
    }

    #[test]
    fn test_binary_cross_entropy() {
        let output = array![[0.7], [0.2], [0.1]];
        let observed = array![[0.0], [1.0], [0.0]];
        let cost_function = CostFunction::BinaryCrossEntropy;
        let cost = cost_function.cost(&output, &observed);
        assert!(cost.is_finite());
    }

    #[test]
    fn test_mse() {
        let output = array![[0.7], [0.2], [0.1]];
        let observed = array![[0.0], [1.0], [0.0]];
        let cost_function = CostFunction::Mse;
        let cost = cost_function.cost(&output, &observed);
        assert!(cost.is_finite());
    }
}
