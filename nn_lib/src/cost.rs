use ndarray::ArrayD;

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
    pub fn cost(&self, output: &ArrayD<f64>, observed: &ArrayD<f64>) -> f64 {
        match self {
            Self::CrossEntropy => {
                let epsilon = 1e-7;
                let clipped_output = output.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));
                let correct_class = observed.iter().position(|&x| x == 1.0).unwrap();
                -f64::ln(clipped_output[[correct_class, 0]])
            }
            Self::BinaryCrossEntropy => {
                let epsilon = 1e-7;
                let clipped_output = output.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));
                -(observed * &clipped_output.mapv(f64::ln)
                    + (1.0 - observed) * &((1.0 - clipped_output).mapv(f64::ln)))
                    .mean()
                    .unwrap()
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
        output: &ArrayD<f64>,
        observed: &ArrayD<f64>,
    ) -> ArrayD<f64> {
        match self {
            // the gradient of the cross entropy with respect to the logits
            // is given by : dc/dz = s - y in vector notation.
            // We use this expression over the one that give dc/ds with s the softmax output
            // because the calculation is easy and that prevent us for back propagating through the
            // softmax function
            // THE CROSS ENTROPY AND SOFTMAX are given considering that the sigmiod / softmax layer
            // has been the last layer so output is already a probability distribution
            Self::CrossEntropy => output - observed,
            Self::BinaryCrossEntropy => output - observed,
            Self::Mse => 2f64 * (output - observed) / output.len() as f64,
        }
    }
}
