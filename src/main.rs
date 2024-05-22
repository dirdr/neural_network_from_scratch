mod app;
mod args;
mod xor;

use args::{Arguments, Exemple, Mode};
use clap::Parser;

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    let cli = Arguments::parse();

    match &cli.mode {
        Mode::Gui(_) => {
            println!("Running in GUI mode");
        }
        Mode::Benchmark(options) => match options.run {
            Exemple::Xor => {
                let net = xor::build_neural_net()?;
                xor::start(net)?;
            }
            Exemple::Mnist => {
                let net = mnist::build_neural_net()?;
                mnist::start(net)?;
            }
        },
    }
    Ok(())
}
