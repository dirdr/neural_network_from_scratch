use ndarray::Array2;
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layer::{ActivationLayer, DenseLayer},
    neural_network::NeuralNetworkBuilder,
    optimizer::GradientDescent,
};

use crate::load_dataset;

pub fn build_network() -> anyhow::Result<()> {
    let data_set = load_dataset()?;

    let network = NeuralNetworkBuilder::new()
        .push(DenseLayer::new(28 * 28, 256, InitializerType::He))
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(256, 10, InitializerType::He))
        .build(GradientDescent::new(0.1), CostFunction::CrossEntropy);

    // todo sincder build network et train
    if let Ok(mut network) = network {
        // for the simple MLP perceptron reshape the input images as a single column matrices of
        // shape (28*28, 1);
        let input_train = data_set.training.0.mapv(|e| e as f64);
        let output_train = data_set.training.1.mapv(|e| e as f64);
        let outermost = input_train.shape()[0];

        let input_train = input_train
            .into_shape((outermost, 28 * 28, 1))
            .expect("failed to reshape the input training images");

        let number_of_class = 10;

        let mut one_hot_encoded: Array2<f64> = Array2::zeros((output_train.len(), number_of_class));

        for (i, &label) in output_train.iter().enumerate() {
            let idx = label as usize;
            one_hot_encoded[(i, idx)] = 1.0;
        }

        let output_train = one_hot_encoded
            .into_shape((output_train.len(), number_of_class, 1))
            .expect("failed to reshape one hot encoded vector");

        network.train_par(input_train, output_train, 1000);
    }

    Ok(())
}
