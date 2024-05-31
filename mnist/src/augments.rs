use std::path::Path;

use image::{imageops, GrayImage, Luma};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use log::{debug, trace};
use ndarray::{Array, Array2, ArrayD};
use rand::Rng;

fn array_to_image(arr: &ArrayD<u8>) -> GrayImage {
    let (width, height) = (arr.shape()[1] as u32, arr.shape()[0] as u32);
    let flat_data: Vec<u8> = arr.iter().cloned().collect();
    GrayImage::from_raw(width, height, flat_data).unwrap()
}

fn image_to_array(img: &GrayImage) -> Array2<u8> {
    let (width, height) = img.dimensions();
    let raw_data = img.as_raw();
    Array::from_shape_vec((height as usize, width as usize), raw_data.clone()).unwrap()
}

fn augment_image(image: &ArrayD<u8>, save: bool, index: usize) -> Array2<u8> {
    let mut rng = rand::thread_rng();
    let mut img = array_to_image(image);

    let angle = rng.gen_range(-10.0..10.0);
    img = rotate_image(&img, angle);

    let (x_shift, y_shift) = (rng.gen_range(-5..=5), rng.gen_range(-5..=5));
    img = shift_image(&img, x_shift, y_shift);

    if save {
        img.save(Path::new(&format!("augmented_image_{}.png", index)))
            .unwrap();
    }

    image_to_array(&img)
}

fn rotate_image(img: &GrayImage, angle: f32) -> GrayImage {
    rotate_about_center(
        img,
        angle.to_radians(),
        Interpolation::Bilinear,
        Luma([0u8]),
    )
}

fn shift_image(img: &GrayImage, x_shift: i32, y_shift: i32) -> GrayImage {
    let (width, height) = img.dimensions();
    let mut shifted_img = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let new_x = (x as i32 + x_shift).max(0).min(width as i32 - 1) as u32;
            let new_y = (y as i32 + y_shift).max(0).min(height as i32 - 1) as u32;
            shifted_img.put_pixel(new_x, new_y, *img.get_pixel(x, y));
        }
    }

    shifted_img
}

pub fn augment_dataset(images: &ArrayD<u8>) -> ArrayD<u8> {
    let num_samples = images.shape()[0];
    let mut augmented_images = Array::zeros(images.raw_dim());

    for i in 0..num_samples {
        trace!("augmenting the sample {}", i);
        let image = images.index_axis(ndarray::Axis(0), i).to_owned();
        // save every 1000 images
        let save = i % 10000 == 0;
        let augmented_image = augment_image(&image, false, i);
        augmented_images
            .index_axis_mut(ndarray::Axis(0), i)
            .assign(&augmented_image);
    }
    augmented_images
}
