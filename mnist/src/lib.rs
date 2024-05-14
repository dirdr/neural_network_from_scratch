use std::{io, path::PathBuf};

use log::debug;

use crate::utils::read_data::{decompress_gz_file, read_data};

mod benchmark;
mod network;
mod utils;

const FILES_NAMES: [&str; 4] = [
    "train-labels-idx1-ubyte.gz",
    "train-images-idx3-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
];

fn load_dataset() -> io::Result<()> {
    let base_path = PathBuf::from("./resources");
    let mut data = vec![];

    for file in FILES_NAMES.iter() {
        let compressed = base_path.join("compressed").join(file);
        let raw = base_path.join("raw").join(&file[0..(file.len() - 3)]);
        decompress_gz_file(&compressed, &raw)?;
        let container = read_data(&raw)?;
        debug!("Data : {:?} has shape : {:?}", raw, &container.shape());
        data.push(container);
    }

    Ok(())
}
