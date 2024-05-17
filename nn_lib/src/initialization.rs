use ndarray::Array2;
use ndarray_rand::{rand_distr::Normal, RandomExt};

pub enum InitializerType {
    He,
    RandomNormal(f64, f64),
}

impl InitializerType {
    /// Initialize a new Matrices and fills it according to the `InitializerType`
    /// # Arguments
    /// * `input_size` - The number of input for the initialized layer
    /// * `shape` - output matrices shape
    pub fn initialize(&self, input_size: usize, shape: (usize, usize)) -> Array2<f64> {
        match self {
            InitializerType::He => {
                let std_dev = (2.0 / input_size as f32).sqrt();
                let normal =
                    Normal::new(0.0, std_dev as f64).expect("can't create normal distribution");
                Array2::random(shape, normal)
            }
            InitializerType::RandomNormal(mean, std_dev) => {
                let normal =
                    Normal::new(*mean, *std_dev).expect("can't create normal distribution");
                Array2::random(shape, normal)
            }
        }
    }
}
