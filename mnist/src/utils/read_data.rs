use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read},
    path::Path,
};

use byteorder::{BigEndian, ReadBytesExt};
use flate2::bufread::GzDecoder;
use ndarray::ArrayD;

pub fn decompress_gz_file<P: AsRef<Path>>(input: P, output: P) -> io::Result<()> {
    let file = File::open(input)?;
    let buf_reader = BufReader::new(file);
    let mut gz = GzDecoder::new(buf_reader);
    let output_file = File::create(output)?;
    let mut buf_writer = BufWriter::new(output_file);

    io::copy(&mut gz, &mut buf_writer)?;

    Ok(())
}

// First 4 bytes are the magic number
// - two first bytes are 0
// - third byte is the data type
// - fourth byte is the number of dimension
// dimensions are given next, each dimension is given by (big endian) 4 bytes
pub fn read_idx_data<P: AsRef<Path> + std::fmt::Debug + Copy>(path: P) -> io::Result<ArrayD<u8>> {
    let mut f = BufReader::new(File::open(path)?);
    let magic_number = f.read_u32::<BigEndian>()?;
    let num_dimension = magic_number & 0xFF;
    let mut shape = vec![];

    for _ in 0..num_dimension {
        shape.push(f.read_u32::<BigEndian>()? as usize);
    }

    let total_size: usize = shape.iter().product();
    let mut data = vec![0u8; total_size];
    f.read_exact(&mut data)?;

    Ok(ArrayD::from_shape_vec(shape, data).unwrap())
}
