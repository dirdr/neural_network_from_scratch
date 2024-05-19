use std::any::Any;

use ndarray::Array2;
use thiserror::Error;

use crate::{activation::Activation, initialization::InitializerType};

#[derive(Error, Debug)]
enum LayerError {
    #[error("Access to stored input of the layer before stored happened")]
    IllegalInputAccess,
}

/// The `Layer` trait need to be implemented by any nn layer
///
// In this library, we use a 'Layer-activation decoupling' paradigm, where we separate the 'Dense'
// layers and the 'activation functions'.
/// Instead of defining the activation function inside the layer's output calculation,
/// a `ActivationLayer` is provided, that you will need to plug just after your layer.
///
/// This serve multiple purpose, the first one is separation of concerns, each layer handle his
/// gradient and forward calculation, the second is to make it easy to have fully connected layer
/// without activation function. The third reason is that we found that more instinctive and
/// natural to implement

/// a layer is defined as input nodes x and output nodes y
/// feed forward calculate the output nodes y with respect to the input nodes x
/// `propagate_backward` is responsible for two things :
/// 1: update trainable parameters (if any) in the layer (weights, bias)
/// 2: return the input gradient with respect to the output gradient,
/// the second role is used to propagate partial derivative backward in the network.
/// the layer inputs have shape (i, 1), layer outputs have shape (j, 1).
pub trait Layer: Send + Sync {
    fn feed_forward(&mut self, input: &Array2<f64>) -> Array2<f64>;

    /// Return the gradient of the cost function with respecct to the layer input values
    /// # Arguments
    /// * `output_gradient` - Output gradient vector shape (j, 1))
    fn propagate_backward(
        &mut self,
        output_gradient: &Array2<f64>,
        // TODO peut être modifier ca avec les réseaux convo on est pas sur que la dimension des
        // paramètres entraibles sont les mêmes
    ) -> Array2<f64>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// TODO comment and explain the use of this
pub trait Trainable {
    fn get_parameters(&self) -> Vec<Array2<f64>>;
    fn get_parameters_mut(&mut self) -> Vec<&mut Array2<f64>>;
    fn get_gradients(&self) -> Vec<Array2<f64>>;
}

/// `Dense` Layer (ie: Fully connected layer)
/// weights matrices follow the conversion output first
/// weights_ji connect output node y_j to input node x_i
/// bias b_i is for the calculation of output node y_i
pub struct DenseLayer {
    /// shape (j, i) matrices of weights, w_ji connect node j to node i.
    weights: Array2<f64>,
    /// shape (j, 1) vector of bias
    bias: Array2<f64>,
    /// input passed during the feed forward step
    // TODO utiliser un Arc pour stocker mon machin
    input: Option<Array2<f64>>,
    // store those for optimizer access (from the trait Trainable)
    weights_gradient: Option<Array2<f64>>,
    biases_gradient: Option<Array2<f64>>,
}

impl DenseLayer {
    /// Create a new `DenseLayer` filling it with random value. see `InitializerType` for
    /// initialization parameters
    pub fn new(input_size: usize, output_size: usize, init: InitializerType) -> Self {
        Self {
            weights: init.initialize(input_size, output_size, (output_size, input_size)),
            bias: init.initialize(input_size, output_size, (output_size, 1)),
            input: None,
            weights_gradient: None,
            biases_gradient: None,
        }
    }
}

impl Layer for DenseLayer {
    /// Calculate the output vector (shape (j, 1))
    /// and store the passed input inside the layer to be used in the backpropagation
    fn feed_forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.input = Some(input.clone());
        self.weights.dot(input) + &self.bias
    }

    /// Update trainable parameters (weights and bias)
    /// and return the input gradient vector (shape (i, 1)).
    /// # Arguments
    /// * `input` - (shape (i, 1))
    /// * `output_gradient` - (shape (j, 1))
    fn propagate_backward(&mut self, output_gradient: &Array2<f64>) -> Array2<f64> {
        // Unwrap the input without cloning
        let input = self
            .input
            .as_ref()
            .expect("access to an unset input inside backpropagation");

        // Compute weight gradients
        let weights_gradient = output_gradient.dot(&input.t());

        // Compute input gradients
        let input_gradient = self.weights.t().dot(output_gradient);

        self.weights_gradient = Some(weights_gradient);
        self.biases_gradient = Some(output_gradient.clone());

        input_gradient
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Trainable for DenseLayer {
    fn get_parameters(&self) -> Vec<Array2<f64>> {
        vec![self.weights.clone(), self.bias.clone()]
    }

    fn get_parameters_mut(&mut self) -> Vec<&mut Array2<f64>> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn get_gradients(&self) -> Vec<Array2<f64>> {
        vec![
            self.weights_gradient
                .as_ref()
                .expect("Illegal access to unset weights gradient")
                .clone(),
            self.biases_gradient
                .as_ref()
                .expect("Illegal access to unset biases gradient")
                .clone(),
        ]
    }
}

/// The `ActivationLayer` apply a activation function to it's input node to yield the output nodes.
pub struct ActivationLayer {
    pub activation: Activation,
    pub input: Option<Array2<f64>>,
}

impl ActivationLayer {
    pub fn from(activation: Activation) -> Self {
        Self {
            activation,
            input: None,
        }
    }
}

impl Layer for ActivationLayer {
    /// apply the activation function to each input (shape (i * 1))
    /// return an output vector of shape (i * 1).
    fn feed_forward(&mut self, input: &Array2<f64>) -> Array2<f64> {
        self.input = Some(input.clone());
        self.activation.apply(input)
    }

    /// return the input gradient with respect to the activation layer output gradient
    /// because the activation layer doesn't have trainable parameters, we don't care about the
    /// learning_rate.
    fn propagate_backward(&mut self, output_gradient: &Array2<f64>) -> Array2<f64> {
        let input = self
            .input
            .as_ref()
            // TODO return appropirate error
            .unwrap_or_else(|| panic!("access to a unset input inside backproapgation"));
        output_gradient * self.activation.apply_derivative(input)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct ConvolutionalLayer {
    
}

impl Layer for ConvolutionalLayer {
}
