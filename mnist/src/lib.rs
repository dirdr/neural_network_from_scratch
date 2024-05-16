use std::{
    io,
    path::{Path, PathBuf},
};

use log::{debug, info};
use ndarray::{Array1, Array3, ArrayD};

use crate::utils::read_data::{decompress_gz_file, read_idx_data};

mod benchmark;
mod mnist_network;
mod utils;

pub use mnist_network::build_network;

/// Store the mnist dataset as (images, labels)
/// for the images, dimensions are 1: number of images, 2: number of raw, 3: number of col
/// for the labels, dimension are 1: number of labels
/// images and labels are organized sequentially with images[i] associated with the label[i]
pub struct MnistData {
    // [60000, 28, 28], [60000]
    pub training: (Array3<u8>, Array1<u8>),
    // [60000, 28, 28], [60000]
    pub test: (Array3<u8>, Array1<u8>),
}

const TRAINING: [&str; 2] = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"];

const TEST: [&str; 2] = ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"];

const TRAIN_LEN: usize = 60000;
const TEST_LEN: usize = 10000;

fn load_file(file_name: &str) -> anyhow::Result<ArrayD<u8>> {
    debug!("Trying to load the file : {}", file_name);
    let base_path = PathBuf::from("mnist/resources");
    let compressed = base_path.join("compressed").join(file_name);
    let file_name = Path::new(file_name)
        .file_stem()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "stem file creation failed"))?;
    let raw = base_path.join("raw").join(
        file_name
            .to_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "osstr conversion failed"))?,
    );
    debug!("constructed file name : {:?}", compressed);
    debug!("constructed file name raw : {:?}", raw);

    decompress_gz_file(&compressed, &raw)?;

    let container = read_idx_data(&raw)?;
    debug!("Data : {:?} has shape : {:?}", raw, &container.shape());
    Ok(container)
}

pub fn load_dataset() -> anyhow::Result<MnistData> {
    // load training data set
    let (training_images, training_labels) = (load_file(TRAINING[0])?, load_file(TRAINING[1])?);
    // load test data set
    let (test_images, test_labels) = (load_file(TEST[0])?, load_file(TEST[1])?);

    // TODO create a "validation" set
    debug!(
        "trying to reshape training images from original shape : {:?}",
        training_images.shape()
    );
    let training_images = training_images.into_shape((TRAIN_LEN, 28, 28))?;
    debug!("trying to reshape training labels");
    let training_labels = training_labels.into_shape(TRAIN_LEN)?;
    debug!("trying to reshape test images");
    let test_images = test_images.into_shape((TEST_LEN, 28, 28))?;
    debug!("trying to reshape test labels");
    let test_labels = test_labels.into_shape(TEST_LEN)?;

    info!("Successfully loaded mnist dataset");

    Ok(MnistData {
        training: (training_images, training_labels),
        test: (test_images, test_labels),
    })
}
