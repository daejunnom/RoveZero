use rz_bindings_core::search_once;
use serde::Deserialize;
use std::io::{self, BufRead, Read, Write};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    position_command: String,
    #[serde(default = "default_limit")]
    simulation_limit: u64,
}
fn default_limit() -> u64 {
    128
}
fn reply(request: Result<Request, String>) -> serde_json::Value {
    match request.and_then(|r| search_once(&r.position_command, r.simulation_limit)) {
        Ok(result) => serde_json::json!({"ok":true,"measurement":result}),
        Err(error) => serde_json::json!({"ok":false,"error":error}),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--direct") && args.len() == 4 {
        let loops: u32 = args[2].parse()?;
        if !(1..=20).contains(&loops) {
            return Err("loops must be 1..=20".into());
        }
        let limit = args[3].parse()?;
        for _ in 0..loops {
            println!(
                "{}",
                reply(Ok(Request {
                    position_command: args[1].clone(),
                    simulation_limit: limit
                }))
            );
        }
        return Ok(());
    }
    if !args.is_empty() {
        return Err("use no args for JSONL CLI, or --direct POSITION LOOPS LIMIT".into());
    }
    println!("{{\"ready\":true,\"schema_version\":1}}");
    io::stdout().flush()?;
    // Read at most 16KiB+1 bytes before any input allocation/JSON expansion.
    let mut reader = io::stdin().lock();
    let mut lines = 0;
    loop {
        let mut bytes = Vec::new();
        let count = std::io::Read::by_ref(&mut reader)
            .take(16 * 1024 + 1)
            .read_until(b'\n', &mut bytes)?;
        if count == 0 {
            break;
        }
        lines += 1;
        if count > 16 * 1024 || lines > 512 {
            return Err("CLI input budget exceeded".into());
        }
        let request = serde_json::from_slice(&bytes).map_err(|e| e.to_string());
        println!("{}", reply(request));
        io::stdout().flush()?;
    }
    Ok(())
}
