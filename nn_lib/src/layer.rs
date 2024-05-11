use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr, RandomExt};
use rand_distr::Normal;

trait Layer {
    fn feed_forward(&self, input: Array1<f64>) -> Array1<f64>;

    fn propagate_backbard(
        &mut self,
        input: Array1<f64>,
        output_gradient: Array2<f64>,
        learning_rate: f64,
    ) -> Array2<f64>;
}

/// `Dense` Layer (ie: Fully connected layer)
/// Fully connected layer have input nodes x and output nodes y
/// weights matrice follow the convension output first
/// weights[j, i] connect output node y_j to input node x_i
/// vector of bias b[i] is the bias for the calculation of output node y_i
pub struct DenseLayer {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl DenseLayer {
    fn new(initializer: InitializerType, input_size: usize, output_size: usize) -> Self {
        match initializer {
            InitializerType::He => {
                let std_dev = (2.0 / input_size as f32).sqrt();
                let normal =
                    Normal::new(0.0, std_dev as f64).expect("can't create normal distribution");
                return Self {
                    weights: Array2::random((output_size, input_size), normal),
                    bias: Array1::random(output_size, normal),
                };
            }
            InitializerType::RandomNormal(mean, std_dev) => {
                let normal = Normal::new(mean, std_dev).expect("can't create normal distribution");
                return Self {
                    weights: Array2::random((output_size, input_size), normal),
                    bias: Array1::random(output_size, normal),
                };
            }
        }
    }
}

impl Layer for DenseLayer {
    /// Calculate the output vector values and return it
    fn feed_forward(&self, input: Array1<f64>) -> Array1<f64> {
        self.weights.dot(&input) + &self.bias
    }

    /// perform back propagation for the current layer
    /// take the output_gradient as input, ie the gradient of the cost function
    /// with respect to each output y value
    /// return the gradient of the cost function with respect to input values
    fn propagate_backbard(
        &mut self,
        input: Array1<f64>,
        output_gradient: Array2<f64>,
        learning_rate: f64,
    ) -> Array2<f64> {
        let weights_gradient = output_gradient.dot(&input.t());
        let input_gradient = self.weights.t().dot(&output_gradient);
        self.bias -= learning_rate * output_gradient;
        self.weights -= learning_rate * weights_gradient;
        input_gradient
    }
}

enum InitializerType {
    He,
    RandomNormal(f64, f64),
}
