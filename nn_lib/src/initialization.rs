use ndarray::ArrayD;
use ndarray_rand::{
    rand_distr::{Normal, Uniform},
    RandomExt,
};

pub enum InitializerType {
    He,
    RandomNormal(f64, f64),
    GlorotUniform,
}

impl InitializerType {
    /// Initialize a new Matrices and fills it according to the `InitializerType`
    /// # Arguments
    /// * `input_size` - The number of input for the initialized layer
    /// * `shape` - output matrices shape
    pub fn initialize(
        &self,
        input_size: usize,
        output_size: usize,
        shape: &[usize],
    ) -> ArrayD<f64> {
        match self {
            InitializerType::He => {
                let std_dev = (2.0 / input_size as f64).sqrt();
                let normal = Normal::new(0.0, std_dev).expect("Can't create normal distribution");
                ArrayD::random(shape, normal)
            }
            InitializerType::RandomNormal(mean, std_dev) => {
                let normal =
                    Normal::new(*mean, *std_dev).expect("Can't create normal distribution");
                ArrayD::random(shape, normal)
            }
            InitializerType::GlorotUniform => {
                let limit = (6.0 / (input_size + output_size) as f64).sqrt();
                let uniform = Uniform::new(-limit, limit);
                ArrayD::random(shape, uniform)
            }
        }
    }
}
