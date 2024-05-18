pub trait Optimizer {
    fn get_learning_rate(&self) -> f64;
    fn step(&mut self) -> f64;
}

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

    fn step(&mut self) -> f64 {
        todo!()
    }
}
