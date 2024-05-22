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
    /// Return a new multi-dimensional array initialized according to the `InitializerType`
    ///
    /// # Arguments
    /// * `fan_in` - The number of input in the layer
    /// * `fan_out` The number of output in the layer
    /// * `shape` - output matrices shape
    pub fn initialize(&self, fan_in: usize, fan_out: usize, shape: &[usize]) -> ArrayD<f64> {
        match self {
            InitializerType::He => {
                let std_dev = (2.0 / fan_in as f64).sqrt();
                let normal = Normal::new(0.0, std_dev).expect("Can't create normal distribution");
                ArrayD::random(shape, normal)
            }
            InitializerType::RandomNormal(mean, std_dev) => {
                let normal =
                    Normal::new(*mean, *std_dev).expect("Can't create normal distribution");
                ArrayD::random(shape, normal)
            }
            InitializerType::GlorotUniform => {
                let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
                let uniform = Uniform::new(-limit, limit);
                ArrayD::random(shape, uniform)
            }
        }
    }
}
