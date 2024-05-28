mod app;
mod args;
mod xor;

use app::Application;
use args::{Arguments, Exemple, Mode};
use clap::Parser;

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    let cli = Arguments::parse();

    match &cli.mode {
        Mode::Gui(_) => {
            let native_options = eframe::NativeOptions::default();
            let mut multilayer_perceptron = mnist::get_neural_net(mnist::network::NetType::Mlp)?;
            let mut convolutional_network = mnist::get_neural_net(mnist::network::NetType::Conv)?;
            mnist::start(&mut multilayer_perceptron)?;
            mnist::start(&mut convolutional_network)?;
            let _ = eframe::run_native(
                "Draw a number",
                native_options,
                Box::new(|cc| Box::new(Application::new(cc, multilayer_perceptron, convolutional_network))),
            );
        }
        Mode::Benchmark(options) => match options.run {
            Exemple::Xor => {
                let net = xor::build_neural_net()?;
                xor::start(net)?;
            }
            Exemple::Mnist => {
                let mut net = mnist::get_neural_net(mnist::network::NetType::Conv)?;
                mnist::start(&mut net)?;
            }
        },
    }
    Ok(())
}
