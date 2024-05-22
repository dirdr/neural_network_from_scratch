use std::any::Any;

use ndarray::{linalg::Dot, ArrayD, Axis, ShapeError};
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
//
/// a layer is defined as input nodes x and output nodes y, and have two main functions,
/// `feed_forward()` and `propagate_backward()`
///
/// Layer implementations in this library support batch processing, (i.e processing more than one
/// data point at once).
/// The convention choosen in the layer implementations is (n, features) where n is the number of
/// sample in the batch
pub trait Layer: Send + Sync {
    fn feed_forward(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError>;

    fn propagate_backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, LayerError>;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait Trainable {
    fn get_parameters(&self) -> Vec<ArrayD<f64>>;

    fn get_parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>>;

    fn get_gradients(&self) -> Vec<ArrayD<f64>>;
}

pub struct DenseLayer {
    weights: ArrayD<f64>,
    bias: ArrayD<f64>,
    last_batch_input: Option<ArrayD<f64>>,
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
            weights: init.initialize(input_size, output_size, &[input_size, output_size]),
            bias: init.initialize(input_size, output_size, &[output_size]),
            last_batch_input: None,
            weights_gradient: None,
            biases_gradient: None,
            input_size,
            output_size,
        }
    }
}

impl Layer for DenseLayer {
    /// Return the output matrices of this `DenseLayer` (shape (n, j)), while storing the input matrices
    /// (shape (n, i))
    ///
    /// where **n** is the number of samples, **j** is the layer output size and **i** is the layer
    /// input size.
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        // TODO faire une fonction feed_forward qui prend juste une référence et une fonction
        // feed_forward_mut qui prend une référence mutable pour sauvegarder le dernier input dans
        // le réseau
        // TODO trouver une solution inplace pour éviter de cloner
        self.last_batch_input = Some(input.clone());

        let batch_size = input.shape()[0];
        let input_2d = input.view().into_shape((batch_size, self.input_size))?;
        let weight_2d = self
            .weights
            .view()
            .into_shape((self.input_size, self.output_size))?;

        Ok((input_2d.dot(&weight_2d) + &self.bias).into_dyn())
    }

    /// Return the input gradient vector (shape (n, i)), by processing the output gradient vector
    /// (shape (n, j)).
    ///
    /// This function also compute and store the current batch weights and biases gradient in the layer.
    ///
    /// # Arguments
    /// * `input` - (shape (n, i))
    /// * `output_gradient` - (shape (n, j))
    fn propagate_backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, LayerError> {
        let input_gradient = match self.last_batch_input.as_ref() {
            Some(input) => {
                let batch_size = output_gradient.shape()[0];
                let output_grad_2d = output_gradient
                    .view()
                    .into_shape((batch_size, self.output_size))?;

                let input_2d = input.view().into_shape((batch_size, self.input_size))?;

                let weight_2d = self
                    .weights
                    .view()
                    .into_shape((self.input_size, self.output_size))?;

                // mean relative to the batch
                let weights_gradient = input_2d.t().dot(&output_grad_2d) / batch_size as f64;
                let biases_gradient = output_grad_2d.sum_axis(Axis(0)) / batch_size as f64;

                self.weights_gradient = Some(weights_gradient.to_owned().into_dyn());
                self.biases_gradient = Some(biases_gradient.into_dyn());

                Ok((output_grad_2d.dot(&weight_2d.t())).into_dyn())
            }
            None => Err(LayerError::IllegalInputAccess),
        };
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

impl Layer for ActivationLayer {
    /// Return a matrice (shape (n, i)) with the activation function applied to a batch
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        self.input = Some(input.clone());
        Ok(self.activation.apply(input))
    }

    /// Return the input gradient (shape (n, i)) of this `ActivationLayer` by processing the output gradient.
    /// # Arguments
    /// * `output_gradient` shape (n, j)
    fn propagate_backward(
        &mut self,
        output_gradient: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>, LayerError> {
        let input_gradient = match self.input.as_ref() {
            Some(input) => {
                // debug!(" input shape {:?}", input.shape());
                // debug!(" ourput gradient shape {:?}", output_gradient.shape());
                Ok(output_gradient * self.activation.apply_derivative(input))
            }
            None => Err(LayerError::IllegalInputAccess),
        };
        input_gradient
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
