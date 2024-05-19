use crate::layer::Trainable;

pub trait Optimizer: Sync + Send {
    fn get_learning_rate(&self) -> f64;
    fn step(&mut self, layer: &mut dyn Trainable);
}

#[derive(Clone)]
pub struct GradientDescent {
    learning_rate: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> Self {
        GradientDescent { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn step(&mut self, layer: &mut dyn Trainable) {
        let gradients = layer.get_gradients();

        let mut parameters = layer.get_parameters_mut();

        for (param, grad) in parameters.iter_mut().zip(gradients.iter()) {
            param.scaled_add(-self.learning_rate, grad);
        }
    }
}
