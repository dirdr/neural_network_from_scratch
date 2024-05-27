use std::any::Any;
use ndarray::{s, ArrayD, ArrayView, ArrayViewMut, Axis, Dimension, Ix3, IxDyn, ShapeError, par_azip};
use thiserror::Error;

use crate::{activation::Activation, initialization::InitializerType};

/// The `Layer` trait need to be implemented by any nn layer
//
/// a layer is defined as input nodes x and output nodes y, and have two main functions,
/// `feed_forward()` and `propagate_backward()`
///
/// Layer implementations in this library support batch processing, (i.e. processing more than one
/// data point at once).
/// The convention chosen in the layer implementations is (n, features) where n is the number of
/// sample in the batch
pub trait Layer {
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError>;

    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError>;

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
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        // TODO find a without clone method, like in place mutation
        self.last_batch_input = Some(input.clone());
        self.feed_forward(input)
    }

    /// Return the output matrices of this `DenseLayer` (shape (n, j))
    ///
    /// where **n** is the number of samples, **j** is the layer output size.
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
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
    /// Return a matrices (shape (n, i)) with the activation function applied to a batch
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        self.input = Some(input.clone());
        self.feed_forward(input)
    }

    /// Return a matrices (shape (n, i)) with the activation function applied to a batch
    /// while storing the input for later use in backpropagation process
    ///
    /// # Arguments
    /// * `input` - shape (n, i)
    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
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
            Some(input) => Ok(output_gradient * self.activation.apply_derivative(input)),
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

pub struct ConvolutionalLayer {
    kernels: ArrayD<f64>,
    bias: ArrayD<f64>,
    input: Option<ArrayD<f64>>,
    kernel_gradient: Option<ArrayD<f64>>,
    bias_gradient: Option<ArrayD<f64>>,

    input_size: (usize, usize, usize),
    output_size: (usize, usize, usize),
    kernels_size: (usize, usize, usize, usize),
}

impl ConvolutionalLayer {
    pub fn new(
        input_size: (usize, usize, usize),
        kernel_size: (usize, usize),
        number_of_kernel: usize,
        init: InitializerType,
    ) -> Self {
        let (kernel_height, kernel_width): (usize, usize) = kernel_size;
        let (input_height, input_width, input_channel): (usize, usize, usize) = input_size;

        let output_size: (usize, usize, usize) = (
            input_height - kernel_height + 1,
            input_width - kernel_width + 1,
            number_of_kernel,
        );
        let (output_height, output_width, output_channel): (usize, usize, usize) = output_size;

        Self {
            kernels: init.initialize(
                input_height * input_width * input_channel,
                output_height * output_width * output_channel,
                &[kernel_height, kernel_width, input_channel, number_of_kernel],
            ),
            bias: init.initialize(
                input_height * input_width * input_channel,
                output_height * output_width * output_channel,
                &[number_of_kernel],
            ),
            input: None,
            kernel_gradient: None,
            bias_gradient: None,
            input_size,
            output_size,
            kernels_size: (kernel_height, kernel_width, input_channel, number_of_kernel),
        }
    }
}

impl Layer for ConvolutionalLayer {
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        self.input = Some(input.clone());
        self.feed_forward(input)
    }

    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        let batch_size: usize = input.shape()[0];
        let input_channel: usize = self.input_size.2;
        let (kernel_height, kernel_width, kernel_depth, number_of_kernel): (usize, usize, usize, usize) = self.kernels_size;
        assert_eq!(input_channel, kernel_depth, "Input depth must match kernel depth");

        let (output_height, output_width, output_depth): (usize, usize, usize) = self.output_size;

        let mut output: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, output_height, output_width, output_depth]));

        for batch_index in 0..batch_size {
            for index_kernel in 0..number_of_kernel {
                let kernel = self.kernels.slice(s![.., .., .., index_kernel]);
                for y in 0..output_height {
                    for x in 0..output_width {
                        let input_slice = input.slice(s![batch_index, y..y + kernel_height, x..x + kernel_width, ..]);
                        let convolution_result = (&input_slice * &kernel).sum() + self.bias[index_kernel];
                        output[[batch_index, y, x, index_kernel]] = convolution_result;
                    }
                }
            }
        }
        Ok(output)
    }

    fn propagate_backward(&mut self, output_gradient: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        let batch_size: usize = output_gradient.shape()[0];
        let (input_height, input_width, input_channel): (usize, usize, usize) = self.input_size;
        let (kernel_height, kernel_width, kernel_depth, number_of_kernel) = self.kernels_size;
        let (output_height, output_width, _): (usize, usize, usize) = self.output_size;

        let mut input_gradient: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, input_height, input_width, input_channel]));
        let mut kernel_gradient: ArrayD<f64> = ArrayD::zeros(IxDyn(&[kernel_height, kernel_width, kernel_depth, number_of_kernel]));
        let mut bias_gradient: ArrayD<f64> = ArrayD::zeros(IxDyn(&[number_of_kernel]));

        let input = self.input.as_ref().unwrap();

        for batch_index in 0..batch_size {
            for index_kernel in 0..number_of_kernel {
                let kernel: ArrayView<f64, Ix3> = self.kernels.slice(s![.., .., .., index_kernel]);
                // Combine y and x loops to reduce slicing overhead
                for (y, x) in (0..output_height).flat_map(|y| (0..output_width).map(move |x| (y, x))) {
                    let gradient: f64 = output_gradient[[batch_index, y, x, index_kernel]];

                    // Calculate slices once
                    let input_slice_range = s![batch_index, y..y + kernel_height, x..x + kernel_width, ..];

                    // Calculate input_gradient
                    {
                        let input_slice = input_gradient.slice_mut(input_slice_range);
                        par_azip!((input_val in input_slice, &kernel_val in kernel) {
                            *input_val += gradient * kernel_val;
                        });
                    }

                    // Calculate kernel_gradient
                    {
                        let input_slice = input.slice(input_slice_range);
                        let kernel_change_slice = kernel_gradient.slice_mut(s![.., .., .., index_kernel]);
                        par_azip!((kernel_change_val in kernel_change_slice, &input_val in input_slice) {
                            *kernel_change_val += gradient * input_val;
                        });
                    }

                    // Update bias_gradient
                    bias_gradient[index_kernel] += gradient;
                }
            }
        }
        
        self.kernel_gradient = Some(kernel_gradient);
        self.bias_gradient = Some(bias_gradient);

        Ok(input_gradient)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Trainable for ConvolutionalLayer {
    fn get_parameters(&self) -> Vec<ArrayD<f64>> {
        vec![
            self.kernels.clone().into_dyn(),
            self.bias.clone().into_dyn(),
        ]
    }

    fn get_parameters_mut(&mut self) -> Vec<&mut ArrayD<f64>> {
        vec![&mut self.kernels, &mut self.bias]
    }

    fn get_gradients(&self) -> Vec<ArrayD<f64>> {
        vec![
            self.kernel_gradient
                .as_ref()
                .expect("Illegal access to unset weights gradient")
                .clone()
                .into_dyn(),
            self.bias_gradient
                .as_ref()
                .expect("Illegal access to unset biases gradient")
                .clone()
                .into_dyn(),
        ]
    }
}

pub struct ReshapeLayer {
    input: Option<ArrayD<f64>>,
    input_shape: IxDyn,
    output_shape: IxDyn,
}

impl ReshapeLayer {
    pub fn new(
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<Self, LayerError> {
            let input_elements: usize = input_shape.iter().product();
            let output_elements: usize = output_shape.iter().product();
            if input_elements != output_elements {
                return Err(LayerError::ReshapeError(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape)));
            }
            Ok(Self {
                input: None,
                input_shape: IxDyn(input_shape),
                output_shape: IxDyn(output_shape),
            })
    }
}

impl Layer for ReshapeLayer {
    fn feed_forward_save(&mut self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        self.input = Some(input.clone());
        self.feed_forward(input)
    }

    fn feed_forward(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        let batch_size: usize = input.shape()[0];
        let mut shape: Vec<usize> = Vec::with_capacity(self.output_shape.ndim() + 1);
        shape.push(batch_size);
        shape.extend_from_slice(self.output_shape.as_array_view().as_slice().unwrap());

        if input.shape().iter().product::<usize>() != shape.iter().product() {
            return Err(LayerError::ReshapeError(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape)));
        }
        Ok(input.clone().into_shape(shape).unwrap())
    }

    fn propagate_backward(&mut self, output_gradient: &ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        let batch_size: usize = output_gradient.shape()[0];
        let mut shape: Vec<usize> = Vec::with_capacity(self.output_shape.ndim() + 1);
        shape.push(batch_size);
        shape.extend_from_slice(self.input_shape.as_array_view().as_slice().unwrap());
        if output_gradient.shape().iter().product::<usize>() != shape.iter().product() {
            return Err(LayerError::ReshapeError(ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape)));
        }
        Ok(output_gradient.clone().into_shape(shape).unwrap())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Error, Debug)]
pub enum LayerError {
    #[error("Access to stored input of the layer before stored happened")]
    IllegalInputAccess,

    #[error("Error reshaping array: {0}")]
    ReshapeError(#[from] ShapeError),

    #[error("Dimension don't match")]
    DimensionMismatch,
}
