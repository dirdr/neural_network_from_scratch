use std::{any::Any, fmt::Display};

use ndarray::{ArrayD, ShapeError};
use thiserror::Error;

use crate::{activation::Activation, initialization::InitializerType};

#[derive(Error, Debug)]
pub enum LayerError {
    #[error("Access to stored input of the layer before stored happened")]
    IllegalInputAccess,

    #[error("Error reshaping array: {0}")]
    ReshapeError(#[from] ShapeError),
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
    fn feed_forward(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError>;

    /// Return the gradient of the cost function with respecct to the layer input values
    /// # Arguments
    /// * `output_gradient` - Output gradient vector shape (j, 1))
    fn propagate_backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, LayerError>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// TODO comment and explain the use of this
pub trait Trainable {
    fn get_parameters(&self) -> Vec<ArrayD<f64>>;

    fn get_parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>>;

    fn get_gradients(&self) -> Vec<ArrayD<f64>>;
}

/// `Dense` Layer (ie: Fully connected layer)
/// weights matrices follow the conversion output first
/// weights_ji connect output node y_j to input node x_i
/// bias b_i is for the calculation of output node y_i
pub struct DenseLayer {
    /// shape (j, i) matrices of weights, w_ji connect node j to node i.
    weights: ArrayD<f64>,
    /// shape (j, 1) vector of bias
    bias: ArrayD<f64>,
    /// input passed during the feed forward step
    // TODO utiliser un Arc pour stocker mon machin
    last_input: Option<ArrayD<f64>>,
    // store those for optimizer access (from the trait Trainable)
    weights_gradient: Option<ArrayD<f64>>,
    biases_gradient: Option<ArrayD<f64>>,
    input_size: usize,
    output_size: usize,
}

impl DenseLayer {
    /// Create a new `DenseLayer` filling it with random value. see `InitializerType` for
    /// initialization parameters
    pub fn new(input_size: usize, output_size: usize, init: InitializerType) -> Self {
        Self {
            weights: init.initialize(input_size, output_size, &[output_size, input_size]),
            bias: init.initialize(input_size, output_size, &[output_size, 1]),
            last_input: None,
            weights_gradient: None,
            biases_gradient: None,
            input_size,
            output_size,
        }
    }
}

impl Layer for DenseLayer {
    /// Calculate the output vector (shape (j, 1))
    /// and store the passed input inside the layer to be used in the backpropagation
    fn feed_forward(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        let input_2d = input.view().into_shape((self.input_size, 1))?;

        let weight_2d = self
            .weights
            .view()
            .into_shape((self.output_size, self.input_size))?;

        self.last_input = Some(input.clone());

        let output = weight_2d.dot(&input_2d) + &self.bias;
        Ok(output.into_dyn())
    }

    /// Return the input gradient vector (shape (i, 1)) by processing the output gradient vector
    /// # Arguments
    /// * `input` - (shape (i, 1))
    /// * `output_gradient` - (shape (j, 1))
    fn propagate_backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, LayerError> {
        // Ensure `self.input` is correctly set from forward propagation
        let input = self
            .last_input
            .as_ref()
            .expect("access to an unset input inside backpropagation");

        // Ensure both input and output_gradient are two-dimensional
        let output_grad_2d = output_gradient.view().into_shape((self.output_size, 1))?;

        let input_2d = input.view().into_shape((self.input_size, 1))?;

        let weight_2d = self
            .weights
            .view()
            .into_shape((self.output_size, self.input_size))?;

        let weights_gradient = output_grad_2d.dot(&input_2d.t());

        let input_gradient = weight_2d.t().dot(&output_grad_2d);

        self.weights_gradient = Some(weights_gradient.to_owned().into_dyn());
        self.biases_gradient = Some(output_grad_2d.to_owned().into_dyn());

        Ok(input_gradient.into_dyn())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Trainable for DenseLayer {
    fn get_parameters(&self) -> Vec<ArrayD<f64>> {
        vec![
            self.weights.clone().into_dyn(),
            self.bias.clone().into_dyn(),
        ]
    }

    fn get_parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn get_gradients(&self) -> Vec<ArrayD<f64>> {
        vec![
            self.weights_gradient
                .as_ref()
                .expect("Illegal access to unset weights gradient")
                .clone()
                .into_dyn(),
            self.biases_gradient
                .as_ref()
                .expect("Illegal access to unset biases gradient")
                .clone()
                .into_dyn(),
        ]
    }
}

/// The `ActivationLayer` apply a activation function to it's input node to yield the output nodes.
pub struct ActivationLayer {
    pub activation: Activation,
    pub input: Option<ArrayD<f64>>,
}

impl ActivationLayer {
    pub fn from(activation: Activation) -> Self {
        Self {
            activation,
            input: None,
        }
    }
}

impl Display for ActivationLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Activationlayer : {}\n", self.activation))
    }
}

impl Layer for ActivationLayer {
    /// apply the activation function to each input (shape (i * 1))
    /// return an output vector of shape (i * 1).
    fn feed_forward(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        self.input = Some(input.clone());
        Ok(self.activation.apply(input))
    }

    /// return the input gradient with respect to the activation layer output gradient
    /// because the activation layer doesn't have trainable parameters, we don't care about the
    /// learning_rate.
    fn propagate_backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, LayerError> {
        let input = self
            .input
            .as_ref()
            .unwrap_or_else(|| panic!("access to a unset input inside backproapgation"));
        Ok(output_gradient * self.activation.apply_derivative(input))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// pub struct ConvolutionalLayer {
//     kernels: Array4<f64>,
//     bias: Array2<f64>,
//     input: Option<Array3<f64>>,
//     kernel_gradient: Option<Array4<f64>>,
//     bias_gradient: Option<Array2<f64>>,
// }
//
// impl ConvolutionalLayer {
//     pub fn new(
//         input_size: (usize, usize, usize),
//         kernel_size: (usize, usize),
//         number_of_kernel: usize,
//         init: InitializerType,
//     ) -> Self {
//         let (kernel_height, kernel_width): (usize, usize) = kernel_size;
//         let (input_height, input_width, input_depth): (usize, usize, usize) = input_size;
//
//         let output_size: (usize, usize, usize) = (
//             input_height - kernel_height + 1,
//             input_width - kernel_width + 1,
//             number_of_kernel,
//         );
//         let (output_height, output_width, output_depth): (usize, usize, usize) = output_size;
//
//         Self {
//             kernels: init.initialize_4d(
//                 input_height * input_width * input_depth,
//                 output_height * output_width * output_depth,
//                 (number_of_kernel, kernel_height, kernel_width, input_depth),
//             ),
//             bias: init.initialize(
//                 input_height * input_width * input_depth,
//                 output_height * output_width * output_depth,
//                 (number_of_kernel, 1),
//             ),
//             input: None,
//             kernel_gradient: None,
//             bias_gradient: None,
//         }
//     }
// }
//
// impl ConvolutionalLayer {
//     fn feed_forward(&mut self, input: &Array3<f64>) -> Array3<f64> {
//         self.input = Some(input.clone());
//         let (number_of_kernel, kernel_height, kernel_width, kernel_depth): (usize, usize, usize, usize) = self.kernels.dim();
//         let (input_height, input_width, input_depth): (usize, usize, usize) = input.dim();
//         assert_eq!(input_depth, kernel_depth, "Input depth must match kernel depth");
//
//         let output_size: (usize, usize, usize) = (
//             input_height - kernel_height + 1,
//             input_width - kernel_width + 1,
//             number_of_kernel
//         );
//         let (output_height, output_width, _): (usize, usize, usize) = output_size;
//
//         let mut output: Array3<f64> = Array3::zeros(output_size);
//
//         for index_kernel in 0..number_of_kernel {
//             let kernel = self.kernels.slice(s![index_kernel, .., .., ..]);
//             for y in 0..output_height {
//                 for x in 0..output_width {
//                     let input_slice = input.slice(s![y..y + kernel_height, x..x + kernel_width, ..]);
//                     let convolution_result = (&input_slice * &kernel).sum() + self.bias[[index_kernel, 1]];
//                     output[[y, x, index_kernel]] = convolution_result.max(0.0);
//                 }
//             }
//         }
//         output
//     }
//
//     fn propagate_backward(&mut self, output_gradient: &Array3<f64>) -> Array3<f64> {
//         let (number_of_kernels, kernel_height, kernel_width, input_depth) = self.kernels.dim();
//         let (input_height, input_width, _) = self.input.as_ref().unwrap().dim();
//         let (output_height, output_width, _): (usize, usize, usize) = output_gradient.dim();
//
//         let mut input_gradient = Array3::<f64>::zeros((input_height, input_width, input_depth));
//
//         let kernel_gradient = self.kernel_gradient.as_mut().expect("Can't modify kernel in conv layer");
//         let bias_gradient = self.bias_gradient.as_mut().expect("Can't modify bias in conv layer");
//         let input = self.input.as_ref().unwrap();
//
//         for index_kernel in 0..number_of_kernels {
//             let kernel = self.kernels.slice(s![index_kernel, .., .., ..]);
//             for y in 0..output_height {
//                 for x in 0..output_width {
//                     if input[[y, x, index_kernel]] <= 0.0 {
//                         continue;
//                     }
//                     let grad = output_gradient[[y, x, index_kernel]];
//
//                     let mut input_slice = input_gradient.slice_mut(s![y..y + kernel_height, x..x + kernel_width, ..]);
//
//                     Zip::from(&mut input_slice)
//                         .and(&kernel)
//                         .for_each(|input_val, &kernel_val| {
//                             *input_val += grad * kernel_val;
//                         });
//
//                     let input_slice = input.slice(s![y..y + kernel_height, x..x + kernel_width, ..]);
//                     let mut kernel_change_slice = kernel_gradient.slice_mut(s![index_kernel, .., .., ..]);
//
//                     Zip::from(&mut kernel_change_slice)
//                         .and(&input_slice)
//                         .for_each(|kernel_change_val, &input_val| {
//                             *kernel_change_val += grad * input_val;
//                         });
//
//                     bias_gradient[[index_kernel, 1]] += grad;
//                 }
//             }
//         }
//
//         // Apply gradients with a learning rate (example: 0.01)
//         let learning_rate = 0.01;
//         self.kernels -= &(kernel_gradient.mapv(|x| x * learning_rate));
//         self.bias -= &(bias_gradient.mapv(|x| x * learning_rate));
//
//         input_gradient
//     }
// }
