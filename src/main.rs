use mnist::load_dataset;

fn main() {
    pretty_env_logger::init();
    let data_set = load_dataset();
}
