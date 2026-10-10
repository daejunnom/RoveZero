//! Feature-independent resource declarations for the explicit resident lane.
//!
//! Strict JSON validation grants no graph/body/namespace/Session/provider/Run
//! authority. Native conversion remains feature-gated and uses the same closed
//! bounds. These are declarations, never measured allocation or VRAM peaks.

const RECORD_CAPACITY: usize = 128;
const MAX_INSPECTION_HOST_BYTES: u64 = 1024 * 1024;
const MAX_WIRE_FIELDS: u64 = 65_536;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceResourceInputError {
    InvalidDeclaration,
    Overflow,
    NativeProfileMismatch,
}
impl std::fmt::Display for DeviceResourceInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "resident resource declaration: {self:?}")
    }
}
impl std::error::Error for DeviceResourceInputError {}
type Result<T> = std::result::Result<T, DeviceResourceInputError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaRecordPagePayloadDeclaration {
    pub host: u64,
    pub device: u64,
}
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaRecordPagesLimitsInput {
    pub max_blocks: usize,
    pub max_bank_whole_payload_bytes: u64,
    pub max_registry_entries: usize,
    pub max_container_host_bytes: u64,
    pub max_invocation_host_bytes: u64,
    pub max_invocation_device_bytes: u64,
}
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaRecordPagesInvocationInput {
    pub public_session_bytes: u64,
    pub packing_session_bytes: u64,
    pub private_session_bytes: u64,
    pub private_output_payload: NativeCudaRecordPagePayloadDeclaration,
    pub original_input_and_transfer_payload: NativeCudaRecordPagePayloadDeclaration,
    pub additional_owner_metadata_payload: NativeCudaRecordPagePayloadDeclaration,
}
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaPackingArtifactResourcesInput {
    pub packing_session_bytes: u64,
    pub additional_owner_metadata_host_bytes: u64,
    pub max_owned_artifact_host_bytes: u64,
    pub max_declared_host_bytes: u64,
    pub max_declared_device_bytes: u64,
}
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCudaPackingGraphInspectionInput {
    pub max_inspection_host_bytes: u64,
    pub max_wire_fields: u64,
}
/// Missing/null/unknown/duplicate/float controls are rejected. There are no
/// defaults or sentinels, and even directly constructed declarations revalidate.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct NativeCudaRecordPagesResourceInput {
    pub limits: NativeCudaRecordPagesLimitsInput,
    pub invocation: NativeCudaRecordPagesInvocationInput,
    pub packing_artifact: NativeCudaPackingArtifactResourcesInput,
    pub graph_inspection: NativeCudaPackingGraphInspectionInput,
}
impl<'de> serde::Deserialize<'de> for NativeCudaRecordPagesResourceInput {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            limits: NativeCudaRecordPagesLimitsInput,
            invocation: NativeCudaRecordPagesInvocationInput,
            packing_artifact: NativeCudaPackingArtifactResourcesInput,
            graph_inspection: NativeCudaPackingGraphInspectionInput,
        }
        let raw = <Raw as serde::Deserialize>::deserialize(deserializer)?;
        let value = Self {
            limits: raw.limits,
            invocation: raw.invocation,
            packing_artifact: raw.packing_artifact,
            graph_inspection: raw.graph_inspection,
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}
impl NativeCudaRecordPagesResourceInput {
    pub fn validate(&self) -> Result<()> {
        use DeviceResourceInputError::{InvalidDeclaration, Overflow};
        let l = &self.limits;
        let i = &self.invocation;
        let a = &self.packing_artifact;
        let g = &self.graph_inspection;
        let max_blocks = RECORD_CAPACITY.checked_add(1).ok_or(Overflow)?;
        let max_registry = max_blocks
            .checked_add(1)
            .and_then(|n| n.checked_mul(max_blocks))
            .ok_or(Overflow)?;
        if l.max_blocks == 0
            || l.max_blocks > max_blocks
            || l.max_registry_entries == 0
            || l.max_registry_entries > max_registry
            || l.max_bank_whole_payload_bytes < u64::try_from(l.max_blocks).map_err(|_| Overflow)?
            || l.max_container_host_bytes == 0
            || l.max_invocation_host_bytes == 0
            || l.max_invocation_device_bytes == 0
            || i.public_session_bytes == 0
            || i.packing_session_bytes == 0
            || i.private_session_bytes == 0
            || a.packing_session_bytes != i.packing_session_bytes
            || a.additional_owner_metadata_host_bytes == 0
            || a.max_owned_artifact_host_bytes == 0
            || a.max_declared_host_bytes == 0
            || a.max_declared_device_bytes == 0
            || g.max_inspection_host_bytes == 0
            || g.max_inspection_host_bytes > MAX_INSPECTION_HOST_BYTES
            || g.max_wire_fields == 0
            || g.max_wire_fields > MAX_WIRE_FIELDS
            || l.max_container_host_bytes > l.max_invocation_host_bytes
            || a.max_declared_host_bytes > l.max_invocation_host_bytes
            || a.max_declared_device_bytes > l.max_invocation_device_bytes
        {
            return Err(InvalidDeclaration);
        }
        let sessions = i
            .public_session_bytes
            .checked_add(i.packing_session_bytes)
            .and_then(|n| n.checked_add(i.private_session_bytes))
            .ok_or(Overflow)?;
        let mut host = 0u64;
        let mut device = sessions;
        for payload in [
            i.private_output_payload,
            i.original_input_and_transfer_payload,
            i.additional_owner_metadata_payload,
        ] {
            if payload.host.checked_add(payload.device).ok_or(Overflow)? == 0 {
                return Err(InvalidDeclaration);
            }
            host = host.checked_add(payload.host).ok_or(Overflow)?;
            device = device.checked_add(payload.device).ok_or(Overflow)?;
        }
        if host > l.max_invocation_host_bytes
            || device > l.max_invocation_device_bytes
            || i.additional_owner_metadata_payload.host < a.additional_owner_metadata_host_bytes
        {
            return Err(InvalidDeclaration);
        }
        let artifact_host = a
            .max_owned_artifact_host_bytes
            .checked_add(a.additional_owner_metadata_host_bytes)
            .ok_or(Overflow)?;
        if artifact_host > a.max_declared_host_bytes
            || a.packing_session_bytes > a.max_declared_device_bytes
        {
            return Err(InvalidDeclaration);
        }
        Ok(())
    }
    #[cfg(all(
        feature = "onnx",
        feature = "contracts",
        feature = "experimental-io-binding"
    ))]
    pub fn validated_parts(
        &self,
    ) -> Result<(
        crate::pals_onnx::DevicePagesLimits,
        crate::pals_onnx::DevicePageInvocationDeclaration,
        crate::pals_onnx::DevicePackingResourceDeclaration,
        crate::pals_onnx::PackingGraphInspectionBudget,
    )> {
        self.validate()?;
        use crate::pals_onnx as native;
        if native::DEVICE_PAGE_RECORD_CAPACITY != RECORD_CAPACITY
            || native::PACKING_GRAPH_MAX_INSPECTION_HOST_BYTES != MAX_INSPECTION_HOST_BYTES
            || native::PACKING_GRAPH_MAX_WIRE_FIELDS != MAX_WIRE_FIELDS
        {
            return Err(DeviceResourceInputError::NativeProfileMismatch);
        }
        let payload = |p: NativeCudaRecordPagePayloadDeclaration| native::DevicePagePayload {
            host: p.host,
            device: p.device,
        };
        Ok((
            native::DevicePagesLimits {
                max_blocks: self.limits.max_blocks,
                max_bank_whole_payload_bytes: self.limits.max_bank_whole_payload_bytes,
                max_registry_entries: self.limits.max_registry_entries,
                max_container_host_bytes: self.limits.max_container_host_bytes,
                max_invocation_host_bytes: self.limits.max_invocation_host_bytes,
                max_invocation_device_bytes: self.limits.max_invocation_device_bytes,
            },
            native::DevicePageInvocationDeclaration {
                public_session_bytes: Some(self.invocation.public_session_bytes),
                packing_session_bytes: Some(self.invocation.packing_session_bytes),
                private_session_bytes: Some(self.invocation.private_session_bytes),
                private_output_payload: Some(payload(self.invocation.private_output_payload)),
                original_input_and_transfer_payload: Some(payload(
                    self.invocation.original_input_and_transfer_payload,
                )),
                additional_owner_metadata_payload: Some(payload(
                    self.invocation.additional_owner_metadata_payload,
                )),
            },
            native::DevicePackingResourceDeclaration {
                packing_session_bytes: Some(self.packing_artifact.packing_session_bytes),
                additional_owner_metadata_host_bytes: Some(
                    self.packing_artifact.additional_owner_metadata_host_bytes,
                ),
                max_owned_artifact_host_bytes: self.packing_artifact.max_owned_artifact_host_bytes,
                max_declared_host_bytes: self.packing_artifact.max_declared_host_bytes,
                max_declared_device_bytes: self.packing_artifact.max_declared_device_bytes,
            },
            native::PackingGraphInspectionBudget {
                max_inspection_host_bytes: self.graph_inspection.max_inspection_host_bytes,
                max_wire_fields: self.graph_inspection.max_wire_fields,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn declaration() -> NativeCudaRecordPagesResourceInput {
        NativeCudaRecordPagesResourceInput {
            limits: NativeCudaRecordPagesLimitsInput {
                max_blocks: 2,
                max_bank_whole_payload_bytes: 4096,
                max_registry_entries: 3,
                max_container_host_bytes: 4096,
                max_invocation_host_bytes: 16384,
                max_invocation_device_bytes: 65536,
            },
            invocation: NativeCudaRecordPagesInvocationInput {
                public_session_bytes: 4096,
                packing_session_bytes: 4096,
                private_session_bytes: 4096,
                private_output_payload: NativeCudaRecordPagePayloadDeclaration {
                    host: 0,
                    device: 4096,
                },
                original_input_and_transfer_payload: NativeCudaRecordPagePayloadDeclaration {
                    host: 1024,
                    device: 1024,
                },
                additional_owner_metadata_payload: NativeCudaRecordPagePayloadDeclaration {
                    host: 4096,
                    device: 0,
                },
            },
            packing_artifact: NativeCudaPackingArtifactResourcesInput {
                packing_session_bytes: 4096,
                additional_owner_metadata_host_bytes: 1024,
                max_owned_artifact_host_bytes: 4096,
                max_declared_host_bytes: 8192,
                max_declared_device_bytes: 8192,
            },
            graph_inspection: NativeCudaPackingGraphInspectionInput {
                max_inspection_host_bytes: MAX_INSPECTION_HOST_BYTES,
                max_wire_fields: MAX_WIRE_FIELDS,
            },
        }
    }
    #[test]
    fn exact_json_round_trip_is_declaration_only() {
        let value = declaration();
        let bytes = serde_json::to_vec(&value).unwrap();
        let decoded: NativeCudaRecordPagesResourceInput = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value, decoded);
        assert!(decoded.validate().is_ok());
    }
    #[test]
    fn rejects_unknown_missing_null_float_boolean_and_zero_fields() {
        for mutation in 0..7 {
            let mut wire = serde_json::to_value(declaration()).unwrap();
            match mutation {
                0 => {
                    wire["limits"]["unknown"] = serde_json::json!(1);
                }
                1 => {
                    wire.as_object_mut().unwrap().remove("graph_inspection");
                }
                2 => wire["invocation"]["public_session_bytes"] = serde_json::Value::Null,
                3 => wire["limits"]["max_blocks"] = serde_json::json!(2.0),
                4 => wire["limits"]["max_blocks"] = serde_json::json!(true),
                5 => wire["packing_artifact"]["packing_session_bytes"] = serde_json::json!(0),
                _ => wire["invocation"]["private_output_payload"]["extra"] = serde_json::json!(1),
            }
            assert!(serde_json::from_value::<NativeCudaRecordPagesResourceInput>(wire).is_err());
        }
    }
    #[test]
    fn duplicate_nested_and_top_level_keys_are_rejected() {
        let original = serde_json::to_string(&declaration()).unwrap();
        let nested = original.replacen("\"max_blocks\":2", "\"max_blocks\":2,\"max_blocks\":2", 1);
        assert!(serde_json::from_str::<NativeCudaRecordPagesResourceInput>(&nested).is_err());
        let limits = serde_json::to_string(&declaration().limits).unwrap();
        let duplicate = format!("{{\"limits\":{limits},{}", &original[1..]);
        assert!(serde_json::from_str::<NativeCudaRecordPagesResourceInput>(&duplicate).is_err());
    }
    #[test]
    fn direct_struct_mutation_rechecks_overflow_and_cross_budget_mismatch() {
        for mutation in 0..6 {
            let mut value = declaration();
            match mutation {
                0 => value.invocation.private_session_bytes = u64::MAX,
                1 => value.invocation.packing_session_bytes += 1,
                2 => {
                    value.invocation.original_input_and_transfer_payload =
                        NativeCudaRecordPagePayloadDeclaration {
                            host: u64::MAX,
                            device: 1,
                        }
                }
                3 => value.limits.max_registry_entries = usize::MAX,
                4 => value.graph_inspection.max_wire_fields += 1,
                _ => value.packing_artifact.max_owned_artifact_host_bytes = u64::MAX,
            }
            assert!(value.validate().is_err());
        }
    }
    #[cfg(all(
        feature = "onnx",
        feature = "contracts",
        feature = "experimental-io-binding"
    ))]
    #[test]
    fn native_conversion_keeps_declarations_without_minting_authority() {
        let value = declaration();
        let (limits, invocation, artifact, inspection) = value.validated_parts().unwrap();
        assert_eq!(
            limits.max_invocation_device_bytes,
            value.limits.max_invocation_device_bytes
        );
        assert_eq!(
            invocation.public_session_bytes,
            Some(value.invocation.public_session_bytes)
        );
        assert_eq!(
            artifact.packing_session_bytes,
            Some(value.packing_artifact.packing_session_bytes)
        );
        assert_eq!(
            inspection.max_wire_fields,
            value.graph_inspection.max_wire_fields
        );
    }
}
