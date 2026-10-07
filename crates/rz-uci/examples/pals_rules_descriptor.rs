//! Finite CPU-only encoder declaration. Redirect stdout to a managed artifact
//! root and pass it to the model wrapper's --rules-profile-json option.
//! A matching declaration does not prove learned-input suitability.
use rz_eval::pals_model::PalsModelConfig;
use rz_uci::pals_native::{
    PALS_DIVERGENCE_FEATURES, PALS_METADATA_FEATURES, PALS_QUERY_FEATURES, PALS_RECORD_FEATURES,
    PALS_RULES_BASE_SEMANTICS, PALS_RULES_ENCODING, pals_native_source_digest,
    pals_rules_encoding_semantic_digest, pals_rules_semantic_fields,
};
use std::fmt::Write as _;

fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch < '\u{20}' => {
                write!(out, "\\u{:04x}", ch as u32).expect("String formatting");
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}
fn array<const N: usize>(items: [&str; N]) -> String {
    strings(items.into_iter())
}
fn strings<'a>(items: impl Iterator<Item = &'a str>) -> String {
    format!("[{}]", items.map(quoted).collect::<Vec<_>>().join(","))
}
fn hex(bytes: [u8; 32]) -> String {
    bytes.into_iter().map(|b| format!("{b:02x}")).collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args_os().skip(1).next().is_some() {
        return Err(
            "pals_rules_descriptor takes no arguments and emits one fixed declaration".into(),
        );
    }
    let c = PalsModelConfig::baseline();
    // No runtime/model loading, training, benchmark, or filesystem inspection.
    let text = format!(
        concat!(
            "{{\"schema\":\"rovezero.pals-rules-descriptor.v1\",",
            "\"rules_input_profile\":{},\"rules_input_semantic_sha256\":{},",
            "\"rules_encoder_source_sha256\":{},\"encoder_source\":\"crates/rz-uci/src/pals_native.rs\",",
            "\"semantic_digest_algorithm\":\"sha256_u64le_length_prefixed_utf8_fields\",",
            "\"rules_base_semantics\":{},\"semantic_fields\":{},",
            "\"learned_input_compatibility\":\"unverified_declaration_only\",",
            "\"metadata_features\":{},\"record_features\":{},\"query_features\":{},\"divergence_features\":{},",
            "\"config\":{{\"width\":{},\"query_heads\":{},\"kv_heads\":{},\"head_dimension\":{},",
            "\"board_blocks\":{},\"record_blocks\":{},\"record_fields\":{},\"latent_slots\":{},",
            "\"recurrent_blocks\":{},\"iterations\":{},\"ffn_width\":{},",
            "\"max_records\":{},\"max_candidates\":{},\"max_divergences\":{}}}}}"
        ),
        quoted(PALS_RULES_ENCODING),
        quoted(&hex(pals_rules_encoding_semantic_digest())),
        quoted(&hex(pals_native_source_digest())),
        quoted(PALS_RULES_BASE_SEMANTICS),
        strings(pals_rules_semantic_fields()),
        array(PALS_METADATA_FEATURES),
        array(PALS_RECORD_FEATURES),
        array(PALS_QUERY_FEATURES),
        array(PALS_DIVERGENCE_FEATURES),
        c.width,
        c.query_heads,
        c.kv_heads,
        c.head_dimension,
        c.board_blocks,
        c.record_blocks,
        c.record_fields,
        c.latent_slots,
        c.recurrent_blocks,
        c.iterations,
        c.ffn_width,
        c.max_records,
        c.max_candidates,
        c.max_divergences
    );
    println!("{text}");
    Ok(())
}
