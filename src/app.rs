use eframe::{App, Frame};
use egui::{epaint::PathShape, CentralPanel, Color32, Context, Painter, Pos2, Rect, Response, Sense, Shape, SidePanel, Stroke, Ui, Vec2, Visuals};
use egui_plot::{Bar, BarChart, Plot};
use image::{GrayImage, ImageBuffer};
use ndarray::{Array2, ArrayD};
use nn_lib::{layer::LayerError, neural_network::NeuralNetwork};

pub struct Application {
    multilayer_perceptron: NeuralNetwork,
    convolutional_network: NeuralNetwork,
    conv_chosen: bool,
    painter_size: Vec2,
    paths: Vec<Vec<Pos2>>,
    current_path: Vec<Pos2>,
    path_shape: PathShape,
    predicted_number: Option<u8>,
}

impl Application {
    pub fn new(creation_context: &eframe::CreationContext<'_>, multilayer_perceptron: NeuralNetwork, convolutional_network: NeuralNetwork) -> Self {
        creation_context.egui_ctx.set_visuals(Visuals::light());
        Self {
            multilayer_perceptron,
            convolutional_network,
            conv_chosen: false,
            painter_size: Vec2::new(280.0, 280.0),
            paths: Vec::default(),
            current_path: Vec::default(),
            path_shape: PathShape {
                points: Vec::default(),
                closed: false,
                fill: Color32::TRANSPARENT,
                stroke: Stroke::new(10.0, Color32::WHITE),
            },
            predicted_number: None,
        }
    }

    fn resize_img_into_28x28(&self) -> anyhow::Result<ArrayD<f64>> {
        let mut img: GrayImage =
            ImageBuffer::from_pixel(self.painter_size.x as u32, self.painter_size.y as u32, image::Luma([0]));
        for path in self.paths.clone() {
            for window in path.windows(2) {
                if let [start, end] = window {
                    self.draw_thick_line(&mut img, *start, *end, 10);
                }
            }
        }
        for window in self.current_path.windows(2) {
            if let [start, end] = window {
                self.draw_thick_line(&mut img, *start, *end, 10)
            }
        }
        let resized_img: GrayImage = image::imageops::resize(&img, 28, 28, image::imageops::FilterType::Lanczos3);
        let _ = resized_img.save("output.png");
        let normalized_pixels: Vec<f64> = resized_img.pixels().map(|p| p[0] as f64 / 255.0).collect();
        let arr = Array2::from_shape_vec((1, 28 * 28), normalized_pixels)?;
        Ok(arr.into_dyn())
    }

    fn predict_number(&mut self, image: ArrayD<f64>) -> Result<ArrayD<f64>, LayerError> {
        if self.conv_chosen {
            self.convolutional_network.predict(&image)
        } else {
            self.multilayer_perceptron.predict(&image)
        }
    }

    fn draw_thick_line(&self, img: &mut GrayImage, start: Pos2, end: Pos2, thickness: i32) {
        for i in -thickness..=thickness {
            for j in -thickness..=thickness {
                self.draw_line(
                    img,
                    Pos2 {
                        x: start.x + i as f32,
                        y: start.y + j as f32,
                    },
                    Pos2 {
                        x: end.x + i as f32,
                        y: end.y + j as f32,
                    },
                );
            }
        }
    }

    fn draw_line(&self, img: &mut GrayImage, start: Pos2, end: Pos2) {
        let start = (start.x as i32, start.y as i32);
        let end = (end.x as i32, end.y as i32);

        let dx = (end.0 - start.0).abs();
        let dy = -(end.1 - start.1).abs();
        let sx = if start.0 < end.0 { 1 } else { -1 };
        let sy = if start.1 < end.1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut x0 = start.0;
        let mut y0 = start.1;

        loop {
            if let Some(pixel) = img.get_pixel_mut_checked(x0 as u32, y0 as u32) {
                *pixel = image::Luma([255]);
            }
            if x0 == end.0 && y0 == end.1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }
}

impl App for Application {
    fn update(&mut self, context: &Context, _frame: &mut Frame) {
        CentralPanel::default().show(context, |ui: &mut Ui| {
            ui.heading("Draw a number");
            ui.heading(if self.conv_chosen { "ConvNet running" } else { "MLP running" });
            
            if ui.button(if self.conv_chosen { "MLP" } else { "ConvNet" }).clicked() {
                self.conv_chosen = !self.conv_chosen;
            }

            let (response, painter): (Response, Painter) = ui.allocate_painter(self.painter_size, Sense::drag());
            let rectangle_painter: Rect = response.rect;
            painter.rect_filled(rectangle_painter, 0.0, Color32::BLACK);

            let left_top_corner_painter: Vec2 = rectangle_painter.min.to_vec2();

            if response.dragged() {
                if let Some(pos) = response.hover_pos() {
                    self.current_path.push(pos - left_top_corner_painter);
                }
            } else if response.drag_stopped() && !self.current_path.is_empty() {
                self.paths.push(self.current_path.clone());
                self.current_path.clear();
            }

            for path in &self.paths {
                painter.add(Shape::Path(PathShape {
                    points: path.iter().map(|point: &Pos2| *point + left_top_corner_painter).collect(),
                    ..self.path_shape
                }));
            }

            if !self.current_path.is_empty() {
                painter.add(Shape::Path(PathShape {
                    points: self.current_path.iter().map(|point: &Pos2| *point + left_top_corner_painter).collect(),
                    ..self.path_shape
                }));
            }


            if ui.button("Clear").clicked() {
                self.current_path.clear();
                self.paths.clear();
                self.predicted_number = None;
            }

            if !self.paths.is_empty() || !self.current_path.is_empty() {
                if let Ok(image) = self.resize_img_into_28x28() {
                    let mut bars = vec![];
                    if let Ok(predictions) = self.predict_number(image) {
                        for (index, prediction) in predictions.iter().enumerate() {
                            let bar: Bar = Bar::new(index as f64, *prediction).name(index);
                            bars.push(bar);
                        }
                    }

                    let bar_chart = BarChart::new(bars).name("Prediction Score").color(egui::Color32::GREEN);
                    Plot::new("Prediction score").view_aspect(2.0).show(ui, |plot_ui| {
                        plot_ui.bar_chart(bar_chart);
                    });
                }
            }
        });
    }
}
