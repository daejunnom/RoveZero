//! External UCI protocol observations do not imply neural/provider attestation.
use crate::{ArenaError, ProcessReceipt};
#[cfg(target_os = "linux")]
use rz_experiments::ExternalUciEndpointV2;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UciOptionDeclaration {
    Spin {
        default: i64,
        min: i64,
        max: i64,
    },
    Check {
        default: bool,
    },
    Combo {
        default: String,
        values: Vec<String>,
    },
    String {
        default: String,
    },
    Button,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalUciAdvertisement {
    pub name: String,
    pub author: Option<String>,
    pub options: BTreeMap<String, UciOptionDeclaration>,
}
#[derive(Clone, Debug, Serialize)]
pub struct ExternalOptionObservation {
    pub requested: String,
    pub advertised: UciOptionDeclaration,
    pub value_supported: bool,
    pub sent_to_preflight: bool,
    pub readiness_barrier_observed: bool,
    /// UCI has no general option readback. readyok is not an applied-value proof.
    pub actual_value: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct ExternalUciPreflight {
    pub schema_version: u32,
    pub engine_id: String,
    pub advertisement: ExternalUciAdvertisement,
    pub options: BTreeMap<String, ExternalOptionObservation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<crate::EngineEnvironmentObservation>,
    pub identification_process: ProcessReceipt,
    pub readiness_process: ProcessReceipt,
    pub stop_and_legal_bestmove_observed: bool,
    pub compiler_information: Vec<String>,
    pub selected_isa: Option<String>,
    pub scope: &'static str,
}
#[derive(Clone, Debug, Serialize)]
pub struct ExternalGameObservation {
    pub engine_id: String,
    pub uci_names: Vec<String>,
    pub uciok_count: u32,
    pub readyok_count: u32,
    pub quit_count: u32,
    pub requested_options_sent_per_game: bool,
    pub actual_options: Option<BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<crate::EngineEnvironmentObservation>,
    pub external_process_ids: Vec<u32>,
    pub process_id_scope: &'static str,
}
fn invalid(reason: &str) -> ArenaError {
    ArenaError::Integrity(format!("external UCI: {reason}"))
}

pub fn parse_uci_advertisement(bytes: &[u8]) -> Result<ExternalUciAdvertisement, ArenaError> {
    if bytes.len() > 256 * 1024 || !bytes.ends_with(b"\n") {
        return Err(invalid(
            "advertisement missing final newline or over budget",
        ));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("advertisement is not UTF-8"))?;
    let mut name = None;
    let mut author = None;
    let mut options = BTreeMap::new();
    let mut complete = false;
    for (index, line) in text.lines().enumerate() {
        if index >= 2048 || line.len() > 4096 || line.chars().any(|c| c.is_control()) {
            return Err(invalid("advertisement line budget/character violation"));
        }
        if complete {
            continue;
        }
        if line == "uciok" {
            complete = true;
            continue;
        }
        if let Some(value) = line.strip_prefix("id name ") {
            if value.is_empty() || value.len() > 256 || name.replace(value.to_string()).is_some() {
                return Err(invalid("duplicate or invalid id name"));
            }
        } else if let Some(value) = line.strip_prefix("id author ") {
            if value.len() > 256 || author.replace(value.to_string()).is_some() {
                return Err(invalid("duplicate or invalid author"));
            }
        } else if let Some(value) = line.strip_prefix("option name ") {
            let (name, body) = value
                .split_once(" type ")
                .ok_or_else(|| invalid("malformed option"))?;
            if name.is_empty() || name.len() > 128 || options.len() >= 128 {
                return Err(invalid("option identity budget"));
            }
            let (kind, tail) = body.split_once(' ').unwrap_or((body, ""));
            let declaration = match kind {
                "spin" => {
                    let tokens: Vec<_> = tail.split_whitespace().collect();
                    if tokens.len() != 6
                        || tokens[0] != "default"
                        || tokens[2] != "min"
                        || tokens[4] != "max"
                    {
                        return Err(invalid("invalid spin bounds"));
                    }
                    let number = |s: &str| {
                        s.parse::<i64>()
                            .map_err(|_| invalid("invalid spin integer"))
                    };
                    let (default, min, max) =
                        (number(tokens[1])?, number(tokens[3])?, number(tokens[5])?);
                    if min > default || default > max {
                        return Err(invalid("spin default outside bounds"));
                    }
                    UciOptionDeclaration::Spin { default, min, max }
                }
                "check" => UciOptionDeclaration::Check {
                    default: match tail {
                        "default true" => true,
                        "default false" => false,
                        _ => return Err(invalid("invalid check default")),
                    },
                },
                "string" => UciOptionDeclaration::String {
                    default: tail
                        .strip_prefix("default ")
                        .or_else(|| (tail == "default").then_some(""))
                        .ok_or_else(|| invalid("missing string default"))?
                        .into(),
                },
                "combo" => {
                    let tail = tail
                        .strip_prefix("default ")
                        .ok_or_else(|| invalid("missing combo default"))?;
                    let mut parts = tail.split(" var ");
                    let default = parts.next().unwrap_or("").to_string();
                    let values: Vec<String> = parts.map(String::from).collect();
                    if values.is_empty()
                        || values.len() > 64
                        || values.iter().any(|v| v.is_empty())
                        || !values.contains(&default)
                    {
                        return Err(invalid("invalid combo alternatives"));
                    }
                    UciOptionDeclaration::Combo { default, values }
                }
                "button" if tail.is_empty() => UciOptionDeclaration::Button,
                _ => return Err(invalid("unsupported option type")),
            };
            if options.insert(name.into(), declaration).is_some() {
                return Err(invalid("duplicate option name"));
            }
        }
    }
    if !complete {
        return Err(invalid("missing uciok"));
    }
    Ok(ExternalUciAdvertisement {
        name: name.ok_or_else(|| invalid("missing id name"))?,
        author,
        options,
    })
}
pub fn validate_uci_options(
    ad: &ExternalUciAdvertisement,
    expected_name: &str,
    requested: &BTreeMap<String, String>,
) -> Result<(), ArenaError> {
    if ad.name != expected_name {
        return Err(invalid("unexpected engine identification"));
    }
    for (name, value) in requested {
        if name.len() > 128
            || name.contains('=')
            || name.chars().any(char::is_control)
            || value.len() > 4096
            || value.chars().any(char::is_control)
        {
            return Err(invalid("invalid option command"));
        }
        let declaration = ad
            .options
            .get(name)
            .ok_or_else(|| invalid("requested option not advertised"))?;
        let valid = match declaration {
            UciOptionDeclaration::Spin { min, max, .. } => value
                .parse::<i64>()
                .is_ok_and(|n| (*min..=*max).contains(&n)),
            UciOptionDeclaration::Check { .. } => matches!(value.as_str(), "true" | "false"),
            UciOptionDeclaration::Combo { values, .. } => values.contains(value),
            UciOptionDeclaration::String { .. } => true,
            UciOptionDeclaration::Button => value.is_empty(),
        };
        if !valid {
            return Err(invalid("requested value outside advertised domain"));
        }
    }
    Ok(())
}
pub fn uci_preflight_commands(
    requested: &BTreeMap<String, String>,
    compiler_probe: bool,
) -> Result<Vec<u8>, ArenaError> {
    let mut text = String::from("uci\n");
    for (name, value) in requested {
        if name.is_empty()
            || name.contains('=')
            || name.chars().any(char::is_control)
            || value.chars().any(char::is_control)
        {
            return Err(invalid("option injection rejected"));
        }
        text.push_str(&format!("setoption name {name}"));
        if !value.is_empty() {
            text.push_str(&format!(" value {value}"));
        }
        text.push('\n');
    }
    text.push_str("isready\nposition startpos\ngo nodes 1\nstop\nisready\n");
    if compiler_probe {
        text.push_str("compiler\n");
    }
    text.push_str("quit\n");
    if text.len() > 64 * 1024 {
        return Err(invalid("protocol input budget exceeded"));
    }
    Ok(text.into_bytes())
}
#[cfg(target_os = "linux")]
pub(crate) fn audit_external_game_protocol(
    stdout: &[u8],
    e: &ExternalUciEndpointV2,
    resolved_options: &BTreeMap<String, String>,
    external_pids: &[u32],
) -> Result<ExternalGameObservation, ArenaError> {
    let text = std::str::from_utf8(stdout).map_err(|_| invalid("runner log is not UTF-8"))?;
    if text.len() > 64 * 1024 * 1024 {
        return Err(invalid("runner trace budget exceeded"));
    }
    let mut names = Vec::new();
    let mut uciok_count = 0;
    let mut readyok_count = 0;
    let mut quit_count = 0;
    let mut sent: BTreeMap<String, u32> = BTreeMap::new();
    for line in text.lines() {
        if !line.starts_with("[Engine]") && !line.starts_with("[ENGINE]") {
            continue;
        }
        let Some((prefix, direction, payload)) = line
            .split_once(" ---> ")
            .map(|(p, v)| (p, true, v))
            .or_else(|| line.split_once(" <--- ").map(|(p, v)| (p, false, v)))
        else {
            continue;
        };
        if !prefix.strip_suffix(&e.id).is_some_and(|p| p.ends_with(' ')) {
            continue;
        }
        match (direction, payload) {
            (true, "uciok") => uciok_count += 1,
            (true, "readyok") => readyok_count += 1,
            (false, "quit") => quit_count += 1,
            (true, p) if p.starts_with("id name ") => names.push(p[8..].to_string()),
            (false, p) if p.starts_with("setoption name ") => {
                *sent.entry(p.into()).or_default() += 1;
            }
            _ => {}
        }
    }
    let options_ok = resolved_options.iter().all(|(k, v)| {
        sent.get(&format!("setoption name {k} value {v}"))
            .is_some_and(|n| *n == 2)
    });
    if names.len() != 2
        || names.iter().any(|n| n != &e.expected_uci_name)
        || uciok_count != 2
        || readyok_count < 2
        || quit_count != 2
        || !options_ok
        || external_pids.len() != 2
    {
        return Err(invalid(
            "two fresh game UCI identities/options/readiness/quit/exit witnesses are incomplete",
        ));
    }
    Ok(ExternalGameObservation {
        engine_id: e.id.clone(),
        uci_names: names,
        uciok_count,
        readyok_count,
        quit_count,
        requested_options_sent_per_game: options_ok,
        actual_options: None,
        environment: e
            .environment
            .as_ref()
            .map(crate::engine_environment::observation),
        external_process_ids: external_pids.to_vec(),
        process_id_scope: "two non-native exit identities in the same supervised four-engine pair; game-to-PID mapping unknown",
    })
}
#[cfg(target_os = "linux")]
pub(crate) fn resolve_asset_tokens(
    value: &str,
    e: &ExternalUciEndpointV2,
    pins: &[crate::native_launch::linux::InputPin],
) -> Result<String, ArenaError> {
    let mut result = value.to_string();
    for (i, asset) in e.assets.iter().enumerate() {
        let token = format!("{{{{asset:{i}}}}}");
        if result.contains(&token) {
            let pin = crate::native_launch::linux::pin(pins, asset)?;
            let path = pin
                .path
                .to_str()
                .ok_or_else(|| invalid("asset path is not UTF-8"))?;
            result = result.replace(&token, path);
        }
    }
    if result.contains("{{asset:") || result.len() > 4096 {
        return Err(invalid("unresolved or oversized asset binding"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    const AD:&[u8]=b"id name Stockfish 19\noption name Threads type spin default 1 min 1 max 1024\noption name Ponder type check default false\noption name NumaPolicy type string default auto\noption name Style type combo default Normal var Normal var Very Solid\nuciok\n";
    #[test]
    fn advertised_domains_and_protocol_injection_are_checked() {
        let ad = parse_uci_advertisement(AD).unwrap();
        let mut options = BTreeMap::from([
            ("Threads".into(), "2".into()),
            ("Ponder".into(), "false".into()),
            ("NumaPolicy".into(), "none".into()),
            ("Style".into(), "Very Solid".into()),
        ]);
        validate_uci_options(&ad, "Stockfish 19", &options).unwrap();
        options.insert("Threads".into(), "1025".into());
        assert!(validate_uci_options(&ad, "Stockfish 19", &options).is_err());
        options.insert("Threads".into(), "2\nquit".into());
        assert!(uci_preflight_commands(&options, false).is_err());
        assert!(parse_uci_advertisement(&AD[..AD.len() - 6]).is_err());
        assert!(
            parse_uci_advertisement(
                b"id name A\noption name T type spin default 2 min 3 max 4\nuciok\n"
            )
            .is_err()
        );
    }
    #[test]
    fn readyok_does_not_invent_actual_option_values() {
        let ad = parse_uci_advertisement(AD).unwrap();
        let observation = ExternalOptionObservation {
            requested: "2".into(),
            advertised: ad.options["Threads"].clone(),
            value_supported: true,
            sent_to_preflight: true,
            readiness_barrier_observed: true,
            actual_value: None,
        };
        assert!(
            serde_json::to_string(&observation)
                .unwrap()
                .contains("\"actual_value\":null")
        );
    }
}
