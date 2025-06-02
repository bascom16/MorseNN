use burn_import::onnx::ModelGen;
use burn_import::burn::graph::RecordType;

fn main() {
    ModelGen::new()
        .input("src/model/morse_model.onnx")
        .out_dir("src/model/")
        .record_type(RecordType::Bincode)
        .embed_states(true)
        .run_from_script();

    println!("cargo:rustc-link-arg-bins=--nmagic");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    #[cfg(feature = "defmt")]
    println!("cargo:rustc-link-arg-bins=-Tdefmt.x");
}
