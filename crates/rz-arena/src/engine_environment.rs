//! Per-engine public environments share the verified input and process lifetime.
#[cfg(target_os = "linux")]
use crate::{
    ArenaError,
    native_launch::linux::{InputPin, pin},
};
#[cfg(target_os = "linux")]
use rz_experiments::EngineEnvironmentV2;
use std::collections::BTreeMap;
#[cfg(target_os = "linux")]
use std::{ffi::OsString, path::Path};

#[derive(Clone, Debug, serde::Serialize)]
pub struct EngineEnvironmentObservation {
    pub launcher_sha256: String,
    pub declared_variables: BTreeMap<String, String>,
    pub clear_inherited_requested: bool,
    pub arguments_prepared: bool,
    /// Launch forwarding is not a readback of engine-specific interpretation.
    pub actual_values: Option<BTreeMap<String, String>>,
    pub scope: &'static str,
}
#[cfg(target_os = "linux")]
pub(crate) fn observation(environment: &EngineEnvironmentV2) -> EngineEnvironmentObservation {
    EngineEnvironmentObservation {
        launcher_sha256: environment.launcher.sha256.clone(),
        declared_variables: environment.variables.clone(),
        clear_inherited_requested: true,
        arguments_prepared: true,
        actual_values: None,
        scope: "pinned clear-environment exec argv; values interpreted by the engine are not read back",
    }
}

#[cfg(target_os = "linux")]
pub(crate) struct EngineProcessLaunch<'a> {
    pub program: &'a InputPin,
    pub arguments: Vec<OsString>,
}

#[cfg(target_os = "linux")]
pub(crate) fn place<'a>(
    mut launch: EngineProcessLaunch<'a>,
    execution: Option<&rz_experiments::MatchExecutionV1>,
    role: rz_experiments::NativeEngineRole,
    clear_environment: bool,
    pins: &'a [InputPin],
) -> Result<EngineProcessLaunch<'a>, ArenaError> {
    let Some(execution) = execution else {
        return Ok(launch);
    };
    let plan = execution.plan()?;
    let allocation = plan
        .engines
        .iter()
        .find(|a| a.role == role)
        .expect("validated role");
    let cpus = allocation
        .cpu_ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let gpus = allocation.gpu_ids.join(",");
    // An inner pinned env -i must preserve the resource mask applied by engine-exec.
    if clear_environment {
        launch
            .arguments
            .insert(4, format!("CUDA_VISIBLE_DEVICES={gpus}").into());
        launch
            .arguments
            .insert(4, "CUDA_DEVICE_ORDER=PCI_BUS_ID".into());
    }
    let gpu_arg = if gpus.is_empty() { "-".into() } else { gpus };
    let mut arguments = vec![
        "engine-exec".into(),
        cpus.into(),
        gpu_arg.into(),
        launch.program.path.as_os_str().to_owned(),
    ];
    arguments.extend(launch.arguments);
    Ok(EngineProcessLaunch {
        program: pin(pins, &execution.executor)?,
        arguments,
    })
}

/// `env` performs exec rather than introducing a shell or persistent proxy.
/// The default branch preserves the prior executable/argv exactly.
#[cfg(target_os = "linux")]
pub(crate) fn prepare<'a>(
    binary: &'a InputPin,
    arguments: Vec<OsString>,
    environment: Option<&EngineEnvironmentV2>,
    variables: Option<&BTreeMap<String, String>>,
    pins: &'a [InputPin],
) -> Result<EngineProcessLaunch<'a>, ArenaError> {
    let Some(environment) = environment else {
        return Ok(EngineProcessLaunch {
            program: binary,
            arguments,
        });
    };
    environment.validate()?;
    let variables = variables.unwrap_or(&environment.variables);
    let resolved = EngineEnvironmentV2 {
        launcher: environment.launcher.clone(),
        variables: variables.clone(),
    };
    resolved.validate()?;
    if variables.keys().ne(environment.variables.keys()) {
        return Err(ArenaError::Integrity(
            "environment resolution changed variable names".into(),
        ));
    }
    Ok(EngineProcessLaunch {
        program: pin(pins, &environment.launcher)?,
        arguments: clear_arguments(&binary.path, arguments, variables),
    })
}

#[cfg(target_os = "linux")]
pub(crate) fn clear_arguments(
    executable: &Path,
    arguments: Vec<OsString>,
    variables: &BTreeMap<String, String>,
) -> Vec<OsString> {
    let mut result = vec![
        "-i".into(),
        "--".into(),
        "LANG=C".into(),
        "PATH=/usr/bin:/bin".into(),
    ];
    result.extend(
        variables
            .iter()
            .map(|(name, value)| OsString::from(format!("{name}={value}"))),
    );
    result.push(executable.as_os_str().to_owned());
    result.extend(arguments);
    result
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use crate::{CleanupStatus, ProcessLimits, ProcessStop};
    use rz_experiments::ArtifactRef;

    #[test]
    fn optional_launch_preserves_arguments_and_requires_the_declared_launcher_pin() {
        // This checks launch selection, not the snapshot's independent digest proof.
        let make_pin = |path: &str, identity: &str| InputPin {
            artifact: ArtifactRef {
                path: identity.into(),
                sha256: "a".repeat(64),
                bytes: 1024,
                source: "https://example.org/public-environment-fixture".into(),
                license: "Synthetic declaration only".into(),
            },
            path: path.into(),
            file: std::fs::File::open(path).unwrap(),
        };
        let binary = make_pin("/usr/bin/printenv", "fixture-engine");
        let launcher = make_pin("/usr/bin/env", "fixture-launcher");
        let environment = EngineEnvironmentV2 {
            launcher: launcher.artifact.clone(),
            variables: BTreeMap::from([(
                "RZ_PUBLIC_FIXTURE".into(),
                "literal $() ; with spaces".into(),
            )]),
        };
        let arguments = vec![
            OsString::from("argument with spaces"),
            OsString::from("last"),
        ];
        let legacy = prepare(&binary, arguments.clone(), None, None, &[]).unwrap();
        assert!(std::ptr::eq(legacy.program, &binary));
        assert_eq!(legacy.arguments, arguments);
        assert!(prepare(&binary, arguments.clone(), Some(&environment), None, &[]).is_err());
        let pins = vec![launcher];
        let declared =
            prepare(&binary, arguments.clone(), Some(&environment), None, &pins).unwrap();
        assert!(std::ptr::eq(declared.program, &pins[0]));
        let encoded = crate::encode_fastchess_native_args(&declared.arguments).unwrap();
        assert!(
            encoded
                .to_str()
                .unwrap()
                .contains("\"RZ_PUBLIC_FIXTURE=literal $() ; with spaces\"")
        );
        assert_eq!(
            &declared.arguments[declared.arguments.len() - 2..],
            &arguments
        );
        let wrong_names = BTreeMap::from([("OTHER_PUBLIC_FIXTURE".into(), "literal".into())]);
        assert!(
            prepare(
                &binary,
                arguments,
                Some(&environment),
                Some(&wrong_names),
                &pins
            )
            .is_err()
        );
        assert!(observation(&environment).actual_values.is_none());
    }

    #[test]
    fn exec_environments_are_isolated_literal_and_owned_until_exit() {
        let env = std::fs::File::open("/usr/bin/env").unwrap();
        let limits = ProcessLimits {
            wall_ms: 2000,
            shutdown_grace_ms: 200,
            max_output_bytes: 4096,
            max_child_processes: 3,
        };
        for value in ["first role $() ; literal value", "second role"] {
            let variables = BTreeMap::from([("RZ_PUBLIC_FIXTURE".into(), value.into())]);
            let arguments = clear_arguments(
                Path::new("/usr/bin/printenv"),
                vec!["RZ_PUBLIC_FIXTURE".into(), "PATH".into(), "HOME".into()],
                &variables,
            );
            let output =
                crate::supervise(&env, &arguments, Path::new("/tmp"), limits, None).unwrap();
            assert_eq!(
                String::from_utf8(output.stdout).unwrap(),
                format!("{value}\n/usr/bin:/bin\n")
            );
            // printenv exits 1 because the inherited HOME is absent.
            assert_eq!(output.receipt.stop, ProcessStop::Exited);
            assert_eq!(output.receipt.exit_code, Some(1));
            assert_eq!(output.receipt.group_cleanup, CleanupStatus::Gone);
            assert!(output.pending_child.is_none());
            assert!(output.receipt.errors.is_empty());
        }
    }
}
