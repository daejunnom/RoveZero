//! Explicit V5 route. Legacy rz-uci flags and startup remain independently usable.
fn main() {
    if let Err(error) = rz_uci::pals_adaptive_bootstrap::run_cli(std::env::args().skip(1).collect())
    {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
