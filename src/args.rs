use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug, Clone)]
#[command(
    name = "neural network from scratch",
    about = "A simple neural network library written in rust",
    author = "Adrien P. <adrien.pelfresne@edu.esiee.fr>, Alexis VAPAILLE. <alexis.vapaille@edu.esiee.fr>",
    version = "1.0.0"
)]
pub struct Arguments {
    #[command(subcommand)]
    pub mode: Mode,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Mode {
    /// Run in GUI mode
    Gui(GuiOptions),

    /// Run benchmarks
    Benchmark(BenchmarkOptions),
}

#[derive(Parser, Debug, Clone)]
pub struct GuiOptions {
    // GUI-specific options can be added here
}

#[derive(Parser, Debug, Clone)]
pub struct BenchmarkOptions {
    #[arg(short, long, default_value = "xor")]
    pub run: Exemple,

    #[arg(short, long)]
    pub epochs: Option<usize>,
}

#[derive(Copy, Clone, ValueEnum, Debug, PartialOrd, Eq, PartialEq)]
pub enum Exemple {
    #[clap(alias = "mnist")]
    Mnist,
    #[clap(alias = "xor")]
    Xor,
}
