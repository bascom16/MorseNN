#![no_std]
#![no_main]

mod fmt;
mod model;

#[cfg(not(feature = "defmt"))]
use panic_halt as _;
#[cfg(feature = "defmt")]
use {defmt_rtt as _, panic_probe as _};

use embassy_executor::Spawner;
use embassy_stm32::gpio::{Level, Output, Speed};
use embassy_time::{Duration, Timer};
use fmt::info;

// heap
use embedded_alloc::LlffHeap as Heap;
#[global_allocator]
static HEAP: Heap = Heap::empty();
// backend
use burn::{backend::NdArray, tensor::Tensor};
type Backend = NdArray<f32>;
type BackendDevice = <Backend as burn::tensor::backend::Backend>::Device;

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    // heap initialization
    {
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 100 * 1024; // This is dependent on the model size in memory.
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(&raw mut HEAP_MEM as usize, HEAP_SIZE) }
    }
    // model
    use model::morse::Model;

    let p = embassy_stm32::init(Default::default());
    let mut led = Output::new(p.PB7, Level::High, Speed::Low);

    loop {
        info!("Hello, World!");
        led.set_high();
        Timer::after(Duration::from_millis(500)).await;
        led.set_low();
        Timer::after(Duration::from_millis(500)).await;
    }
}
