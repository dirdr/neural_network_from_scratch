mod app;
mod args;
mod xor;
mod benchmark;

use app::Application;
use args::{ArgsNetType, Arguments, Exemple, Mode, BenchmarkOptions};
use clap::Parser;
use mnist::network_definition::NetType;
use candle_core::Device;
use log::info;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    let cli = Arguments::parse();

    match &cli.mode {
        Mode::Gui(options) => {
            let native_options = eframe::NativeOptions::default();
            let device = Device::cuda_if_available(0)?;

            let mut multilayer_perceptron = mnist::get_neural_net(NetType::Mlp, &device)?;

            let mut convolutional_perceptron = if options.with_conv {
                Some(mnist::get_neural_net(NetType::Conv, &device)?)
            } else {
                None
            };

            mnist::start(&mut multilayer_perceptron, 128, 10, options.augment, &device)?;

            if let Some(ref mut cnn) = convolutional_perceptron {
                mnist::start(cnn, 128, 10, options.augment, &device)?
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
        Mode::Benchmark(options) => {
            let device = benchmark::get_device(options.device)?;

            if options.compare_devices {
                // Run CPU vs GPU comparison
                use log::info;
                info!("Running CPU vs GPU benchmark comparison");
                run_device_comparison(options)?;
            } else {
                // Run with specified device
                match options.run {
                    Exemple::Xor => {
                        let net = xor::build_neural_net()?;
                        xor::start(net)?;
                    }
                    Exemple::Mnist => {
                        let net_type = match options.net_type {
                            ArgsNetType::Mlp => NetType::Mlp,
                            ArgsNetType::Conv => NetType::Conv,
                        };
                        let mut net = mnist::get_neural_net(net_type, &device)?;
                        mnist::start(&mut net, 128, options.epochs.unwrap_or(10), false, &device)?;
                    }
                }
            }
        },
    }
    Ok(())
}

fn run_device_comparison(options: &BenchmarkOptions) -> anyhow::Result<()> {
    let epochs = options.epochs.unwrap_or(5);
    let batch_size = 128;

    info!("\n{}", "=".repeat(80));
    info!("Starting CPU vs GPU Benchmark Comparison");
    info!("Network types: MLP and Conv");
    info!("Epochs: {}", epochs);
    info!("Batch size: {}", batch_size);
    info!("{}\n", "=".repeat(80));

    // MLP on CPU
    info!("1/4: Training MLP on CPU...");
    let cpu_device = Device::Cpu;
    let start = Instant::now();
    let mut mlp_cpu = mnist::get_neural_net(NetType::Mlp, &cpu_device)?;
    mnist::start(&mut mlp_cpu, batch_size, epochs, false, &cpu_device)?;
    let mlp_cpu_time = start.elapsed().as_secs_f64();
    info!("MLP on CPU completed in {:.3} seconds\n", mlp_cpu_time);

    // MLP on GPU
    info!("2/4: Training MLP on GPU...");
    let gpu_device = Device::new_cuda(0)?;
    let start = Instant::now();
    let mut mlp_gpu = mnist::get_neural_net(NetType::Mlp, &gpu_device)?;
    mnist::start(&mut mlp_gpu, batch_size, epochs, false, &gpu_device)?;
    let mlp_gpu_time = start.elapsed().as_secs_f64();
    info!("MLP on GPU completed in {:.3} seconds\n", mlp_gpu_time);

    // Conv on CPU
    info!("3/4: Training Conv on CPU...");
    let start = Instant::now();
    let mut conv_cpu = mnist::get_neural_net(NetType::Conv, &cpu_device)?;
    mnist::start(&mut conv_cpu, batch_size, epochs, false, &cpu_device)?;
    let conv_cpu_time = start.elapsed().as_secs_f64();
    info!("Conv on CPU completed in {:.3} seconds\n", conv_cpu_time);

    // Conv on GPU
    info!("4/4: Training Conv on GPU...");
    let start = Instant::now();
    let mut conv_gpu = mnist::get_neural_net(NetType::Conv, &gpu_device)?;
    mnist::start(&mut conv_gpu, batch_size, epochs, false, &gpu_device)?;
    let conv_gpu_time = start.elapsed().as_secs_f64();
    info!("Conv on GPU completed in {:.3} seconds\n", conv_gpu_time);

    // Print comparison
    info!("\n{}", "=".repeat(80));
    info!("BENCHMARK RESULTS");
    info!("{}\n", "=".repeat(80));

    info!("{:<30} | {:>20} | {:>20}", "Network", "CPU Time (s)", "GPU Time (s)");
    info!("{:-<30}-+-{:-<20}-+-{:-<20}", "", "", "");
    info!("{:<30} | {:>20.3} | {:>20.3}", "MLP", mlp_cpu_time, mlp_gpu_time);
    info!("{:<30} | {:>20.3} | {:>20.3}", "Conv", conv_cpu_time, conv_gpu_time);
    info!("");
    info!("{:<30} | {:>20.2}x | {:>20.2}x",
        "GPU Speedup",
        mlp_cpu_time / mlp_gpu_time,
        conv_cpu_time / conv_gpu_time
    );
    info!("\n{}\n", "=".repeat(80));

    Ok(())
}
