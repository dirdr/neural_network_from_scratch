use candle_core::Device;
use crate::args::DeviceType;
use std::time::Instant;
use log::info;

pub fn get_device(device_type: DeviceType) -> anyhow::Result<Device> {
    match device_type {
        DeviceType::Auto => Ok(Device::cuda_if_available(0)?),
        DeviceType::Cpu => Ok(Device::Cpu),
        DeviceType::Gpu => {
            if !candle_core::utils::cuda_is_available() {
                anyhow::bail!("GPU requested but CUDA is not available");
            }
            Ok(Device::new_cuda(0)?)
        }
    }
}

pub fn device_name(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "GPU (CUDA)".to_string(),
        Device::Metal(_) => "GPU (Metal)".to_string(),
    }
}

pub struct BenchmarkTimer {
    start: Instant,
    device_name: String,
}

impl BenchmarkTimer {
    pub fn start(device: &Device) -> Self {
        info!("Starting benchmark on {}", device_name(device));
        Self {
            start: Instant::now(),
            device_name: device_name(device),
        }
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    pub fn finish(&self) -> f64 {
        let elapsed = self.elapsed_secs();
        info!("Benchmark on {} completed in {:.3} seconds", self.device_name, elapsed);
        elapsed
    }
}
