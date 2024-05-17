mod xor;

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    let net = xor::build_neural_net()?;
    xor::start(net)?;
    Ok(())
}
