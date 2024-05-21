use clap::{command, Parser, ValueEnum};

#[derive(Parser, Debug, Clone)]
#[command(
    name = "neural network from scratch",
    about = "A simple neural network library written in rust",
    author = "Adrien P. <adrien.pelfresne@edu.esiee.fr>, Alexis VAPAILLE. <alexis.vapaille@edu.esiee.fr>",
    version = "1.0.0"
)]
pub struct Arguments {
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
