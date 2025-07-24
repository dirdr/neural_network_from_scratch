mod args;
mod xor;
mod websocket;

use args::{ArgsNetType, Arguments, Exemple, Mode};
use clap::Parser;
use mnist::network_definition::NetType;
use websocket::WebSocketServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    let cli = Arguments::parse();

    match &cli.mode {
        Mode::Websocket(options) => {
            let mut multilayer_perceptron = mnist::get_neural_net(NetType::Mlp)?;
            
            let mut convolutional_perceptron = if options.with_conv {
                Some(mnist::get_neural_net(NetType::Conv)?)
            } else {
                None
            };

            // Train the models
            mnist::start(&mut multilayer_perceptron, 128, 10, false)?;
            
            if let Some(ref mut cnn) = convolutional_perceptron {
                mnist::start(cnn, 128, 10, false)?;
            }

            // Create and start WebSocket server
            let server = WebSocketServer::new(multilayer_perceptron, convolutional_perceptron);
            println!("Starting WebSocket server on {}", options.address);
            if let Err(e) = server.start(&options.address).await {
                eprintln!("WebSocket server error: {}", e);
            }
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
