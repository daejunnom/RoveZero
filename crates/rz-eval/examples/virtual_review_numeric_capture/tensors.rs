//! Exact V3 prepared host tensors and the independent V5 captured-byte frame.
//! No native runtime, neural execution, or arena dependency exists here.
use rz_eval::pals_model::virtual_review::PalsVirtualReviewInput;
use rz_eval::pals_model::PalsRole;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::error::Error;

const DOMAIN: &str = "rz-pals-captured-tensor-bytes-v5/1";
// Matches the existing V5 numerical raw tensor boundary without introducing
// an rz-eval -> rz-arena/rz-experiments dependency.
const MAX_TENSOR_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TensorCapture {
    pub(super) name: String,
    /// Serialized labels are f32_le/i64_le/bool_u8; framed hash labels remain
    /// the registered f32/i64/bool names rather than Rust or JSON type names.
    pub(super) dtype: String,
    pub(super) shape: Vec<u64>,
    pub(super) bytes: Vec<u8>,
}

pub(super) fn capture(
    input: &PalsVirtualReviewInput,
) -> Result<Vec<TensorCapture>, Box<dyn Error>> {
    let prepared = input.prepare()?;
    let (a, r, c, d, l) = (
        prepared.anchors,
        prepared.records_count,
        prepared.candidates_count,
        prepared.divergences_count,
        prepared.length_bucket,
    );
    let mut tensors = Vec::with_capacity(17);
    add_i64(
        &mut tensors,
        "anchor_board",
        &[1, a, 64],
        &prepared.anchor_board,
    );
    add_bool(&mut tensors, "anchor_mask", &[1, a], &prepared.anchor_mask);
    add_f32(
        &mut tensors,
        "anchor_metadata",
        &[1, a, 16],
        &prepared.anchor_metadata,
    );
    add_bool(
        &mut tensors,
        "candidate_mask",
        &[1, c],
        &prepared.candidate_mask,
    );
    add_i64(&mut tensors, "candidates", &[1, c, 3], &prepared.candidates);
    add_f32(
        &mut tensors,
        "divergence_features",
        &[1, d, 8],
        &prepared.divergence_features,
    );
    add_bool(
        &mut tensors,
        "divergence_mask",
        &[1, d],
        &prepared.divergence_mask,
    );
    add_f32(&mut tensors, "query", &[1, 8], &prepared.query);
    add_i64(
        &mut tensors,
        "query_anchor",
        &[1, 3],
        &prepared.query_anchor,
    );
    add_bool(
        &mut tensors,
        "query_line_mask",
        &[1, 3, l],
        &prepared.query_line_mask,
    );
    add_i64(
        &mut tensors,
        "query_line_tokens",
        &[1, 3, l, 3],
        &prepared.query_line_tokens,
    );
    add_i64(
        &mut tensors,
        "record_anchor",
        &[1, r],
        &prepared.record_anchor,
    );
    add_bool(
        &mut tensors,
        "record_line_mask",
        &[1, r, l],
        &prepared.record_line_mask,
    );
    add_i64(
        &mut tensors,
        "record_line_tokens",
        &[1, r, l, 3],
        &prepared.record_line_tokens,
    );
    add_bool(&mut tensors, "record_mask", &[1, r], &prepared.record_mask);
    add_i64(
        &mut tensors,
        "record_relations",
        &[1, r, 2],
        &prepared.record_relations,
    );
    add_f32(&mut tensors, "records", &[1, r, 8], &prepared.records);
    // role_is_critic is a native dispatch scalar, not one of the registered
    // 17 semantic inputs captured by the V5 numerical protocol.
    tensors.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    digest(&tensors, input.role)?;
    Ok(tensors)
}

/// Exactly matches the independent captured-byte protocol in numerical.rs.
/// It is deliberately distinct from the model's role-dependent input/cache key.
pub(super) fn digest(tensors: &[TensorCapture], role: PalsRole) -> Result<String, Box<dyn Error>> {
    if !(1..=32).contains(&tensors.len()) {
        return Err(invalid("numeric tensor roster bound"));
    }
    let mut hash = Sha256::new();
    frame(&mut hash, DOMAIN.as_bytes());
    let role = match role {
        PalsRole::Proposer => "proposer",
        PalsRole::Critic => "critic",
        PalsRole::Validator => "validator",
    };
    frame(&mut hash, role.as_bytes());
    let mut previous: Option<&str> = None;
    for tensor in tensors {
        if !valid_id(&tensor.name)
            || previous.is_some_and(|name| name >= tensor.name.as_str())
            || tensor.shape.is_empty()
            || tensor.shape.len() > 4
        {
            return Err(invalid(
                "numeric tensors require unique sorted names and bounded shapes",
            ));
        }
        previous = Some(&tensor.name);
        let elements = tensor
            .shape
            .iter()
            .try_fold(1u64, |count, dimension| {
                if *dimension > 0 {
                    count.checked_mul(*dimension)
                } else {
                    None
                }
            })
            .ok_or_else(|| invalid("numeric tensor shape overflow/zero"))?;
        let (label, width) = match tensor.dtype.as_str() {
            "f32_le" => ("f32", 4u64),
            "i64_le" => ("i64", 8u64),
            "bool_u8" => ("bool", 1u64),
            _ => return Err(invalid("numeric tensor dtype is not registered")),
        };
        if elements.checked_mul(width) != Some(tensor.bytes.len() as u64)
            || tensor.bytes.len() > MAX_TENSOR_BYTES
        {
            return Err(invalid("numeric tensor dtype/shape/byte length mismatch"));
        }
        if (tensor.dtype == "f32_le"
            && tensor.bytes.chunks_exact(4).any(|bytes| {
                !f32::from_le_bytes(bytes.try_into().expect("validated f32 width")).is_finite()
            }))
            || (tensor.dtype == "bool_u8" && tensor.bytes.iter().any(|value| *value > 1))
        {
            return Err(invalid("nonfinite numeric input or invalid mask byte"));
        }
        frame(&mut hash, tensor.name.as_bytes());
        frame(&mut hash, label.as_bytes());
        hash.update((tensor.shape.len() as u64).to_le_bytes());
        for dimension in &tensor.shape {
            hash.update(dimension.to_le_bytes());
        }
        frame(&mut hash, &tensor.bytes);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn invalid(detail: &'static str) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, detail).into()
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn frame(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn add_i64(out: &mut Vec<TensorCapture>, name: &str, shape: &[usize], values: &[i64]) {
    out.push(TensorCapture {
        name: name.into(),
        dtype: "i64_le".into(),
        shape: shape.iter().map(|dimension| *dimension as u64).collect(),
        bytes: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    });
}

fn add_f32(out: &mut Vec<TensorCapture>, name: &str, shape: &[usize], values: &[f32]) {
    out.push(TensorCapture {
        name: name.into(),
        dtype: "f32_le".into(),
        shape: shape.iter().map(|dimension| *dimension as u64).collect(),
        bytes: values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    });
}

fn add_bool(out: &mut Vec<TensorCapture>, name: &str, shape: &[usize], values: &[bool]) {
    out.push(TensorCapture {
        name: name.into(),
        dtype: "bool_u8".into(),
        shape: shape.iter().map(|dimension| *dimension as u64).collect(),
        bytes: values.iter().map(|value| u8::from(*value)).collect(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rz_eval::pals_model::virtual_review::{
        ReviewAnchorInput, ReviewQueryInput, ReviewQueryPurpose, INPUT_SCHEMA,
    };
    use rz_eval::pals_model::PalsCandidateToken;

    fn input() -> PalsVirtualReviewInput {
        let mut metadata = [0.; 16];
        metadata[0] = 1.;
        PalsVirtualReviewInput {
            schema: INPUT_SCHEMA.into(),
            role: PalsRole::Proposer,
            anchors: vec![ReviewAnchorInput {
                board: vec![0; 64],
                metadata,
            }],
            records: Vec::new(),
            query: ReviewQueryInput {
                purpose: ReviewQueryPurpose::ProposePolicy,
                perspective: 1,
                prefix: Vec::new(),
                proposal: Vec::new(),
                counter: Vec::new(),
                counter_present: false,
                anchors: [0; 3],
            },
            candidates: vec![PalsCandidateToken {
                from: 12,
                to: 28,
                promotion: 0,
            }],
            divergences: Vec::new(),
            token_budget: 512,
        }
    }
    fn one(name: &str, dtype: &str, shape: Vec<u64>, bytes: Vec<u8>) -> TensorCapture {
        TensorCapture {
            name: name.into(),
            dtype: dtype.into(),
            shape,
            bytes,
        }
    }

    #[test]
    fn prepared_capture_has_exact_seventeen_sorted_inputs_without_dispatch_scalar() {
        let tensors = capture(&input()).unwrap();
        assert_eq!(tensors.len(), 17);
        assert_eq!(tensors[0].name, "anchor_board");
        assert_eq!(tensors[16].name, "records");
        assert!(tensors.iter().all(|tensor| tensor.name != "role_is_critic"));
        assert_eq!(
            tensors
                .iter()
                .find(|tensor| tensor.name == "query_line_tokens")
                .unwrap()
                .shape,
            vec![1, 3, 8, 3]
        );
        assert_eq!(
            tensors
                .iter()
                .find(|tensor| tensor.name == "record_mask")
                .unwrap()
                .bytes,
            vec![0]
        );
        assert_eq!(
            tensors
                .iter()
                .find(|tensor| tensor.name == "record_relations")
                .unwrap()
                .bytes,
            [-1i64, -1]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert!(digest(&tensors, PalsRole::Proposer).is_ok());
    }

    #[test]
    fn captured_host_bytes_preserve_signed_zero_and_use_registered_serialized_dtype() {
        let mut value = input();
        let positive = capture(&value).unwrap();
        value.anchors[0].metadata[1] = -0.;
        let negative = capture(&value).unwrap();
        assert_ne!(positive, negative);
        assert_ne!(
            digest(&positive, PalsRole::Proposer).unwrap(),
            digest(&negative, PalsRole::Proposer).unwrap()
        );
        let tensor = negative
            .iter()
            .find(|tensor| tensor.name == "anchor_metadata")
            .unwrap();
        assert_eq!(&tensor.bytes[4..8], &(-0f32).to_le_bytes());
        assert_eq!(serde_json::to_value(tensor).unwrap()["dtype"], "f32_le");
    }

    #[test]
    fn digest_rejects_noncanonical_order_duplicate_name_or_unregistered_dtype() {
        let first = one("a", "bool_u8", vec![1], vec![1]);
        let second = one("b", "f32_le", vec![1], 1f32.to_le_bytes().to_vec());
        assert!(digest(&[first.clone(), second.clone()], PalsRole::Proposer).is_ok());
        assert!(digest(&[second, first.clone()], PalsRole::Proposer).is_err());
        assert!(digest(&[first.clone(), first], PalsRole::Proposer).is_err());
        assert!(digest(
            &[one("query", "f32", vec![1], 0f32.to_le_bytes().to_vec())],
            PalsRole::Proposer
        )
        .is_err());
        assert!(digest(
            &[one("not a name", "bool_u8", vec![1], vec![1])],
            PalsRole::Proposer
        )
        .is_err());
    }

    #[test]
    fn digest_rejects_invalid_shape_byte_width_nonfinite_or_mask_payload() {
        for shape in [
            Vec::new(),
            vec![0],
            vec![u64::MAX, 2],
            vec![1, 1, 1, 1, 1],
            vec![2],
        ] {
            assert!(digest(
                &[one("query", "f32_le", shape, 0f32.to_le_bytes().to_vec())],
                PalsRole::Proposer
            )
            .is_err());
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(digest(
                &[one(
                    "query",
                    "f32_le",
                    vec![1],
                    value.to_le_bytes().to_vec()
                )],
                PalsRole::Proposer
            )
            .is_err());
        }
        assert!(digest(
            &[one("query", "f32_le", vec![1], vec![0; 3])],
            PalsRole::Proposer
        )
        .is_err());
        assert!(digest(
            &[one("mask", "bool_u8", vec![1], vec![2])],
            PalsRole::Proposer
        )
        .is_err());
        assert!(digest(
            &[one("index", "i64_le", vec![1], vec![0; 4])],
            PalsRole::Proposer
        )
        .is_err());
        assert!(digest(&[], PalsRole::Proposer).is_err());
    }

    #[test]
    fn identical_prepared_tensors_have_separate_private_role_capture_digests() {
        let mut value = input();
        let proposer = capture(&value).unwrap();
        value.role = PalsRole::Critic;
        let critic = capture(&value).unwrap();
        value.role = PalsRole::Validator;
        let validator = capture(&value).unwrap();
        assert_eq!(proposer, critic);
        assert_eq!(proposer, validator);
        let p = digest(&proposer, PalsRole::Proposer).unwrap();
        let c = digest(&critic, PalsRole::Critic).unwrap();
        let v = digest(&validator, PalsRole::Validator).unwrap();
        assert_ne!(p, c);
        assert_ne!(p, v);
        assert_ne!(c, v);
    }
}
