//! 원 source/native/teacher owner를 같은 Collection caller로 연결하는 진입점.
//! 실제 ELF pin과 프로세스/자원 경계는 공유 UCI launch가 별도로 검사한다.
#[cfg(all(target_os = "linux", feature = "pals-collection-onnx"))]
fn main() -> std::process::ExitCode {
    use rz_arena::rovent_collect::launch::{
        ROVENT_COLLECTION_USAGE, RoventCollectionCliParse, parse_rovent_collection_cli,
        prepare_rovent_utility_cohort_pins_from_cli, run_rovent_collection_from_cli,
        run_rovent_utility_cohort_from_cli,
    };
    use std::io::Write;
    let run = || -> Result<Option<serde_json::Value>, Box<dyn std::error::Error>> {
        match parse_rovent_collection_cli(std::env::args().skip(1).collect())? {
            RoventCollectionCliParse::Help => {
                writeln!(std::io::stdout().lock(), "{ROVENT_COLLECTION_USAGE}")?;
                writeln!(
                    std::io::stdout().lock(),
                    "선택형 cohort: --utility-cohort-plan=inputs/<plan.json> --utility-cohort-plan-sha256=<SHA> --utility-cohort-plan-bytes=<bytes> <원 source/config 인수>"
                )?;
                writeln!(
                    std::io::stdout().lock(),
                    "읽기 전용 예상 pin 준비: --prepare-utility-cohort-pins=inputs/<plan.json> --prepare-utility-cohort-pins-sha256=<SHA> --prepare-utility-cohort-pins-bytes=<bytes>"
                )?;
                Ok(None)
            }
            RoventCollectionCliParse::PrepareCohortPins { plan } => {
                Ok(Some(prepare_rovent_utility_cohort_pins_from_cli(&plan)?))
            }
            RoventCollectionCliParse::Cohort { plan, arguments } => {
                let receipt = run_rovent_utility_cohort_from_cli(&plan, arguments)?;
                let resources = receipt.resource_observations()?;
                let collection = receipt.collection_observations()?;
                Ok(Some(
                    serde_json::json!({"schema":"rz-rovent-utility-cohort-launch-result/1",
                    "publication":receipt.publication,"durable_ack":receipt.durable_ack,
                    "root":receipt.output_directory,"trials":receipt.trials,"pairs":receipt.pairs,
                    "resource":resources,"collection":collection,"training_executed":false}),
                ))
            }
            RoventCollectionCliParse::Run { plan, arguments } => {
                let receipt = run_rovent_collection_from_cli(&plan, arguments)?;
                let resource = receipt.resource_observation()?;
                let r_data = receipt
                    .r_datasets
                    .iter()
                    .map(|dataset| dataset.observation())
                    .collect::<Vec<_>>();
                Ok(Some(
                    serde_json::json!({"schema":"rz-rovent-collection-launch-result/1",
                    "publication":receipt.publication_file,"root":receipt.output_directory,
                    "roots":receipt.roots,"records":receipt.records,
                    "teacher_failure_count":receipt.teacher_failure_count,
                    "historical_data":receipt.historical_dataset.observation(),
                    "resource":resource,"r_data":r_data,"r_sidecars":receipt.r_sidecars,
                    "utility_root_local_files":receipt.utility_root_local_files,
                    "training_executed":false}),
                ))
            }
        }
    };
    match run() {
        Ok(result) => {
            if let Some(result) = result
                && writeln!(std::io::stderr().lock(), "{result}").is_err()
            {
                return std::process::ExitCode::FAILURE;
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
