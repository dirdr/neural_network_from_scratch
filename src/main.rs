fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    mnist::build_network()?;
    Ok(())
}
