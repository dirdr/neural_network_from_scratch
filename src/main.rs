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
            let mlp_model_path = "/app/models/mlp_model.json";
            let cnn_model_path = "/app/models/cnn_model.json";
            
            // Try to load existing MLP model or create/train a new one
            let mut multilayer_perceptron = if std::path::Path::new(mlp_model_path).exists() {
                println!("Loading existing MLP model...");
                match nn_lib::sequential::Sequential::load(mlp_model_path) {
                    Ok(builder) => builder.compile(
                        nn_lib::optimizer::GradientDescent::new(0.01),
                        nn_lib::cost::CostFunction::CrossEntropy
                    )?,
                    Err(e) => {
                        println!("Error loading MLP model: {}, training new one...", e);
                        let mut net = mnist::get_neural_net(NetType::Mlp)?;
                        mnist::start(&mut net, 128, 10, false)?;
                        // Save the trained model
                        std::fs::create_dir_all("/app/models").ok();
                        if let Err(e) = net.save(mlp_model_path) {
                            println!("Warning: Could not save MLP model: {}", e);
                        } else {
                            println!("MLP model saved to {}", mlp_model_path);
                        }
                        net
                    }
                }
            } else {
                println!("Training new MLP model...");
                let mut net = mnist::get_neural_net(NetType::Mlp)?;
                mnist::start(&mut net, 128, 10, false)?;
                // Save the trained model
                std::fs::create_dir_all("/app/models").ok();
                if let Err(e) = net.save(mlp_model_path) {
                    println!("Warning: Could not save MLP model: {}", e);
                } else {
                    println!("MLP model saved to {}", mlp_model_path);
                }
                net
            };
            
            let mut convolutional_perceptron = if options.with_conv {
                // Try to load existing CNN model or create/train a new one
                if std::path::Path::new(cnn_model_path).exists() {
                    println!("Loading existing CNN model...");
                    match nn_lib::sequential::Sequential::load(cnn_model_path) {
                        Ok(builder) => Some(builder.compile(
                            nn_lib::optimizer::GradientDescent::new(0.01),
                            nn_lib::cost::CostFunction::CrossEntropy
                        )?),
                        Err(e) => {
                            println!("Error loading CNN model: {}, training new one...", e);
                            let mut net = mnist::get_neural_net(NetType::Conv)?;
                            mnist::start(&mut net, 128, 10, false)?;
                            // Save the trained model
                            if let Err(e) = net.save(cnn_model_path) {
                                println!("Warning: Could not save CNN model: {}", e);
                            } else {
                                println!("CNN model saved to {}", cnn_model_path);
                            }
                            Some(net)
                        }
                    }
                } else {
                    println!("Training new CNN model...");
                    let mut net = mnist::get_neural_net(NetType::Conv)?;
                    mnist::start(&mut net, 128, 10, false)?;
                    // Save the trained model
                    if let Err(e) = net.save(cnn_model_path) {
                        println!("Warning: Could not save CNN model: {}", e);
                    } else {
                        println!("CNN model saved to {}", cnn_model_path);
                    }
                    Some(net)
                }
            } else {
                None
            };

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
