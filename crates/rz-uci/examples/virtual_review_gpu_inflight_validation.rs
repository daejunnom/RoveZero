//! Explicit opt-in; the ordinary UCI command path has no trace.
#[path = "virtual_review_warm_numeric_capture.rs"]
mod shared;
fn main() {
    shared::gpu_inflight_validation::main();
}
