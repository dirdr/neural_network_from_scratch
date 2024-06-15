use ndarray::{ArrayD, Axis};

#[derive(Copy, Clone)]
pub enum Loss {
    CrossEntropy,
    BinaryCrossEntropy,
    Mse,
}

impl Loss {
    /// This crate don't use any kind of auto diff mechanism,
    /// thus, for function like BinaryCrossEntropy and CrossEntropy that need clamped output,
    /// we assume Sigmoid and Softmax respectively as the output activation layer.
    /// the gradient calculation is done with those activation function in mind.
    /// Those function are called 'Output dependant' to contrast with function like Mse, of which
    /// the derivative can be easily calculated with respect to any output layer, because it
    /// doesn't need clamped output.
    pub fn is_output_dependant(&self) -> bool {
        match self {
            Self::BinaryCrossEntropy | Self::CrossEntropy => true,
            Self::Mse => false,
        }
    }

    /// Compute the mean loss of the neural network over a batch, by comparing the network `output` and the
    /// `observed` (label) vector.
    /// Both `output` and `observed` are given as a batch of shape (n, j).
    /// Where `n` is the batch size and `j` is the vector size.
    ///
    /// # Arguments
    /// * `output` - a batch matrice (shape (n, j)) of network outputs
    /// * `observed` - a batch matrice (shape (n, j)) of one hot encoded vector of observed values
    pub fn loss(&self, output: &ArrayD<f64>, observed: &ArrayD<f64>) -> f64 {
        let epsilon = 1e-7;
        let clipped_output = output.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));
        match self {
            Self::CrossEntropy => {
                observed
                    .axis_iter(Axis(0))
                    .enumerate()
                    .map(|(i, observed_row)| {
                        let correct_class = observed_row.iter().position(|&x| x == 1.0).unwrap();
                        -f64::ln(clipped_output[[i, correct_class]])
                    })
                    .sum::<f64>()
                    / output.shape()[0] as f64
            }
            Self::BinaryCrossEntropy => {
                let losses = observed * &clipped_output.mapv(f64::ln)
                    + &(1.0 - observed) * &((1.0 - clipped_output).mapv(f64::ln));
                -losses.mean().unwrap()
            }
            Self::Mse => {
                let diff = output - observed;
                diff.mapv(|x| x.powi(2)).mean().unwrap()
            }
        }
    }

    /// Return the gradient of the loss function with respect to `output` batched matrice.
    /// Note that we don't use thus :
    /// `BinaryCrossEntropy` calculation assume Sigmoid activation as the last layer.
    /// `CrossEntropy` calculation assume Softmax activation as the layer
    ///
    /// # Arguments
    /// * `output` - a batch matrices of neural network outputs (shape (n, j))
    /// * `observed` - a batch matrices of observed values (shape (n, j))
    /// where `n` is the batch_size and `j` is the vector size.
    pub fn loss_output_gradient(
        &self,
        output: &ArrayD<f64>,
        observed: &ArrayD<f64>,
    ) -> ArrayD<f64> {
        match self {
            Self::CrossEntropy => output - observed,
            Self::BinaryCrossEntropy => output - observed,
            Self::Mse => {
                let batch_size = output.shape()[0];
                2f64 * (output - observed) / batch_size as f64
            }
        }
    }
}
