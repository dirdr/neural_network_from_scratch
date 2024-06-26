mod app;
mod args;
mod xor;

use app::Application;
use args::{ArgsNetType, Arguments, Exemple, Mode};
use clap::Parser;
use mnist::network_definition::NetType;

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    let cli = Arguments::parse();

    match &cli.mode {
        Mode::Gui(options) => {
            let native_options = eframe::NativeOptions::default();

            let mut multilayer_perceptron = mnist::get_neural_net(NetType::Mlp)?;

            let mut convolutional_perceptron = if options.with_conv {
                Some(mnist::get_neural_net(NetType::Conv)?)
            } else {
                None
            };

            mnist::start(&mut multilayer_perceptron, 128, 10, options.augment)?;

            if let Some(ref mut cnn) = convolutional_perceptron {
                mnist::start(cnn, 128, 10, options.augment)?
            }

            eframe::run_native(
                "Draw a number",
                native_options,
                Box::new(|cc| {
                    Box::new(Application::new(
                        cc,
                        multilayer_perceptron,
                        convolutional_perceptron,
                    ))
                }),
            )
            .unwrap();
        }
        Mode::Benchmark(options) => match options.run {
            Exemple::Xor => {
                let net = xor::build_neural_net()?;
                xor::start(net)?;
            }
            Exemple::Mnist => {
                let net_type = match options.net_type {
                    ArgsNetType::Mlp => NetType::Mlp,
                    ArgsNetType::Conv => NetType::Conv,
                };
                let mut net = mnist::get_neural_net(net_type)?;
                mnist::start(&mut net, 128, 10, false)?;
            }
        },
    }
    Ok(())
}
