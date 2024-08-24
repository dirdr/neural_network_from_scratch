use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, Default)]
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

#[derive(Subcommand, Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum Mode {
    /// Run in GUI mode
    Gui(GuiOptions),

    /// Run benchmarks
    Benchmark(BenchmarkOptions),
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Gui(GuiOptions::default())
    }
}

#[derive(Parser, Debug, Clone, Hash, PartialEq, Default, PartialOrd, Copy, Ord, Eq)]
pub struct GuiOptions {
    #[arg(short, long, default_value = "false")]
    pub augment: bool,

    #[arg(short, long, default_value = "false")]
    pub with_conv: bool,
}

#[derive(Parser, Debug, Clone, PartialEq, Default, PartialOrd, Copy, Ord, Eq, Hash)]
pub struct BenchmarkOptions {
    #[arg(short, long, default_value = "xor")]
    pub run: Exemple,

    #[arg(short, long)]
    pub epochs: Option<usize>,

    #[arg(short, long, default_value = "mlp")]
    pub net_type: ArgsNetType,
}

#[derive(Copy, Clone, ValueEnum, Debug, PartialOrd, Eq, PartialEq, Ord, Hash, Default)]
pub enum ArgsNetType {
    #[clap(alias = "mlp")]
    #[default]
    Mlp,
    #[clap(alias = "conv")]
    Conv,
}

#[derive(Copy, Clone, ValueEnum, Debug, PartialOrd, Eq, PartialEq, Ord, Default, Hash)]
pub enum Exemple {
    #[clap(alias = "mnist")]
    #[default]
    Mnist,
    #[clap(alias = "xor")]
    Xor,
}
