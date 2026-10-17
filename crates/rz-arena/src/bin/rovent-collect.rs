//! 원 source/native/teacher owner를 같은 Collection caller로 연결하는 진입점.
//! 실제 ELF pin과 프로세스/자원 경계는 공유 UCI launch가 별도로 검사한다.
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn main() -> std::process::ExitCode {
    use rz_arena::rovent_collect::launch::{
        ROVENT_COLLECTION_USAGE, RoventCollectionCliParse, parse_rovent_collection_cli,
        run_rovent_collection_from_cli,
    };
    use std::io::Write;
    let run = || -> Result<Option<serde_json::Value>, Box<dyn std::error::Error>> {
        match parse_rovent_collection_cli(std::env::args().skip(1).collect())? {
            RoventCollectionCliParse::Help => {
                writeln!(std::io::stdout().lock(), "{ROVENT_COLLECTION_USAGE}")?;
                Ok(None)
            }
            RoventCollectionCliParse::Run { plan, arguments } => {
                let receipt = run_rovent_collection_from_cli(&plan, arguments)?;
                Ok(Some(
                    serde_json::json!({"schema":"rz-rovent-collection-launch-result/1",
                    "publication":receipt.publication_file,"root":receipt.output_directory,
                    "roots":receipt.roots,"records":receipt.records,
                    "teacher_failure_count":receipt.teacher_failure_count,
                    "historical_data":receipt.historical_dataset.observation(),
                    "training_executed":false}),
                ))
            }
        }
    };
    match run() {
        Ok(result) => {
            if let Some(result) = result {
                if writeln!(std::io::stderr().lock(), "{result}").is_err() {
                    return std::process::ExitCode::FAILURE;
                }
            }
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            let detail: String = error
                .to_string()
                .chars()
                .take(2048)
                .map(|value| if value.is_control() { ' ' } else { value })
                .collect();
            let _ = writeln!(std::io::stderr().lock(), "rovent-collect halted: {detail}");
            std::process::ExitCode::FAILURE
        }
    }
}
#[cfg(not(all(target_os = "linux", feature = "pals-collection-onnx")))]
fn main() -> std::process::ExitCode {
    use std::io::Write;
    let _ = writeln!(
        std::io::stderr().lock(),
        "rovent-collect requires Linux and pals-collection-onnx"
    );
    std::process::ExitCode::FAILURE
}
