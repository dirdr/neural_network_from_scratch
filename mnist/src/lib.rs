use std::{
    fs, io,
    path::{Path, PathBuf},
};

use log::{debug, info};
use ndarray::{Array1, Array2, Array3, ArrayD};

use crate::utils::read_data::{decompress_gz_file, read_idx_data};

mod benchmark;
mod mnist_network;
mod utils;

pub use mnist_network::*;

/// for the images, dimensions are 1: number of images, 2: number of raw, 3: number of col
/// for the labels, dimension are 1: number of labels
/// images and labels are organized sequentially with images[i] associated with the label[i]
pub struct MnistData {
    // [60000, 28, 28], [60000]
    pub training: (ArrayD<u8>, ArrayD<u8>),
    // [60000, 28, 28], [60000]
    pub test: (ArrayD<u8>, ArrayD<u8>),
}

const TRAINING: [&str; 2] = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"];
const TEST: [&str; 2] = ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"];

const TRAIN_LEN: usize = 60000;
const TEST_LEN: usize = 10000;

fn load_file(file_name: &str) -> anyhow::Result<ArrayD<u8>> {
    debug!("Trying to load the file : {}", file_name);
    let base_path = PathBuf::from("mnist/resources");

    // Ensure the compressed directory exists
    let compressed_dir = base_path.join("compressed");
    fs::create_dir_all(&compressed_dir).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to create compressed directory: {}", e),
        )
    })?;

    let compressed = compressed_dir.join(file_name);
    let file_stem = Path::new(file_name)
        .file_stem()
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "stem file creation failed"))?;

    // Ensure the raw directory exists
    let raw_dir = base_path.join("raw");
    fs::create_dir_all(&raw_dir).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to create raw directory: {}", e),
        )
    })?;

    let raw = raw_dir.join(
        file_stem
            .to_str()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "osstr conversion failed"))?,
    );

    decompress_gz_file(&compressed, &raw)?;

    let container = read_idx_data(&raw)?;
    debug!("Data : {:?} has shape : {:?}", raw, &container.shape());
    Ok(container)
}

pub fn load_dataset() -> anyhow::Result<MnistData> {
    let (training_images, training_labels) = (load_file(TRAINING[0])?, load_file(TRAINING[1])?);
    let (test_images, test_labels) = (load_file(TEST[0])?, load_file(TEST[1])?);

    info!("Successfully loaded mnist dataset");

    Ok(MnistData {
        training: (training_images, training_labels),
        test: (test_images, test_labels),
    })
}
