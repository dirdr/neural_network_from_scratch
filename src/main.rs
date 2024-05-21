mod app;
mod args;
mod xor;

use args::{Arguments, Exemple};
use clap::Parser;

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    let cli = Arguments::parse();

    match cli.run {
        Exemple::Xor => {
            let net = xor::build_neural_net()?;
            xor::start(net)?;
        }
        Exemple::Mnist => todo!(),
    }
    Ok(())
}
