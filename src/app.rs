use egui::{CentralPanel, Context, Pos2};

#[derive(Default)]
pub struct App {
    points: Vec<Pos2>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| ui.label("Hello World!"));
    }
}
