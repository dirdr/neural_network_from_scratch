use eframe::NativeOptions;
use egui::{epaint::PathShape, Color32, Pos2, Shape, Stroke, Vec2, Visuals};
use egui::{CentralPanel, Context, Style};
use image::GrayImage;
use image::ImageBuffer;

#[derive(Default)]
pub struct App {
    points: Vec<Pos2>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| ui.label("Hello World!"));
    }
}

fn app() {
    let painter_size = Vec2::splat(280.0);
    let mut paths: Vec<Vec<Pos2>> = Vec::new();
    let mut current_path: Vec<Pos2> = Vec::new();
    let _ = eframe::run_simple_native(
        "My Simple App",
        NativeOptions::default(),
        Box::new(move |ctx: &egui::Context, _frame: &mut eframe::Frame| {
            let style = Style::default();
            let mut style = style;
            style.visuals = Visuals::light();
            ctx.set_style(style);
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("Draw a number");
                let (response, painter) = ui.allocate_painter(painter_size, egui::Sense::drag());
                let rect = response.rect;
                painter.rect_filled(rect, 0.0, Color32::BLACK);
                if response.dragged() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        current_path.push(pos - rect.min.to_vec2());
                    }
                } else if response.drag_stopped() && !current_path.is_empty() {
                    paths.push(current_path.clone());
                    current_path.clear();
                }

                for path in &paths {
                    painter.add(Shape::Path(PathShape {
                        points: path.iter().map(|p| *p + rect.min.to_vec2()).collect(),
                        closed: false,
                        fill: Color32::TRANSPARENT,
                        stroke: Stroke::new(20.0, Color32::WHITE),
                    }));
                }

                if !current_path.is_empty() {
                    painter.add(Shape::Path(PathShape {
                        points: current_path
                            .iter()
                            .map(|p| *p + rect.min.to_vec2())
                            .collect(),
                        closed: false,
                        fill: Color32::TRANSPARENT,
                        stroke: Stroke::new(10.0, Color32::WHITE),
                    }));
                }

                if ui.button("Export to 28x28").clicked() {
                    export_to_image(&paths, painter_size);
                }

                if ui.button("Clear").clicked() {
                    current_path.clear();
                    paths.clear();
                }
            });
        }),
    );
}

fn draw_thick_line(img: &mut GrayImage, start: Pos2, end: Pos2, thickness: i32) {
    for i in -thickness..=thickness {
        for j in -thickness..=thickness {
            draw_line(
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

fn draw_line(img: &mut GrayImage, start: Pos2, end: Pos2) {
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
        let pixel = img.get_pixel_mut(x0 as u32, y0 as u32);
        *pixel = image::Luma([255]);
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

fn export_to_image(paths: &Vec<Vec<Pos2>>, size: Vec2) {
    let mut img: GrayImage =
        ImageBuffer::from_pixel(size.x as u32, size.y as u32, image::Luma([0]));
    for path in paths {
        for window in path.windows(2) {
            if let [start, end] = window {
                draw_thick_line(&mut img, *start, *end, 10);
            }
        }
    }

    let resized_img = image::imageops::resize(&img, 28, 28, image::imageops::FilterType::Lanczos3);

    resized_img.save("output.png").unwrap();
}
