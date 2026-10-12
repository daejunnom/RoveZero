//! Concrete body inspection for the single weight-free public routing graph.
//!
//! The wire tags follow ONNX v1.17.0 `onnx.proto3`. This is a closed graph
//! inspector, not a general protobuf/ONNX implementation. Owned registered bytes
//! are consumed, inspected, and retained without a second graph buffer. Neither
//! registration authenticity nor native Session/provider/numerical/Run/fence
//! authority is minted. All of those require separate native-owner gates.
//!
//! Inspection reservations cover known backing capacities and a conservative
//! fixed scratch reservation. They are not observed stack/process peak memory,
//! ORT allocator accounting, or CUDA residency. Synthetic wire tests below are
//! not Python artifact, ONNX checker, ORT, or CUDA execution evidence.

use super::device_packing_admission::{
    PackingArtifactRegistration, PackingNativeVerification, RegisteredPackingArtifactBytes,
    DEVICE_PACKING_DOMAIN, DEVICE_PACKING_MAX_GRAPH_BYTES,
};
use sha2::{Digest as _, Sha256};
use std::{fmt, mem::size_of};

pub const PACKING_GRAPH_BODY_INSPECTOR_VERSION: &str = "pals-fixed-packing-graph-body/1";
pub const PACKING_GRAPH_MAX_INSPECTION_HOST_BYTES: u64 = 1024 * 1024;
pub const PACKING_GRAPH_MAX_WIRE_FIELDS: u64 = 65_536;
const MAX_DEPTH: u8 = 8;
const FIXED_SCRATCH_RESERVATION: u64 = 16 * 1024;
const NODES: usize = 275;
const INPUTS: usize = 387;
const OUTPUTS: usize = 3;
const VALUES: usize = 272;
const INITIALIZERS: usize = 5;
const FLOAT: u64 = 1;
const INT64: u64 = 7;
const BOOL: u64 = 9;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingGraphInspectionBudget {
    pub max_inspection_host_bytes: u64,
    /// Counts protobuf fields and each packed numeric element, not just nodes.
    pub max_wire_fields: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackingGraphBodyVerification {
    FixedWeightFreeGraphBodyChecked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingGraphBodyInspection {
    pub graph_sha256: [u8; 32],
    pub graph_bytes: u64,
    pub wire_fields_and_packed_elements: u64,
    /// Known inspection backing capacities plus fixed scratch reservation only.
    pub inspection_host_reserved_bytes: u64,
    pub node_count: usize,
    pub input_count: usize,
    pub output_count: usize,
    pub value_info_count: usize,
    pub integer_control_initializer_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackingGraphDataType {
    Float32,
    Int64,
    Bool,
}
impl PackingGraphDataType {
    fn from_code(code: u64) -> Result<Self> {
        match code {
            FLOAT => Ok(Self::Float32),
            INT64 => Ok(Self::Int64),
            BOOL => Ok(Self::Bool),
            _ => Err(PackingGraphBodyError::FixedGraphMismatch(
                "descriptor dtype",
            )),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingGraphTensorDescriptor {
    pub name: PackingGraphName,
    pub dtype: PackingGraphDataType,
    pub shape: PackingGraphShape,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingGraphNodeDescriptor {
    pub name: PackingGraphName,
    pub operation: &'static str,
    pub output: PackingGraphTensorDescriptor,
    pub input_count: usize,
    pub attribute_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackingGraphBodyError {
    InvalidBudget,
    ArtifactBounds,
    ByteIdentityMismatch,
    MalformedWire(&'static str),
    UnsupportedField { message: &'static str, tag: u32 },
    FixedGraphMismatch(&'static str),
    ResourceExceeded(&'static str),
    Overflow,
    Allocation,
}
impl fmt::Display for PackingGraphBodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fixed packing graph inspection: {self:?}")
    }
}
impl std::error::Error for PackingGraphBodyError {}
type Result<T> = std::result::Result<T, PackingGraphBodyError>;

/// Body proof tied to the same immutable registered manifest/graph pair.
/// There is no mutable byte access, external constructor, verifier callback,
/// native handle, or execution-success flag.
pub struct CheckedFixedPackingGraph {
    artifacts: RegisteredPackingArtifactBytes,
    inspection: PackingGraphBodyInspection,
}
impl fmt::Debug for CheckedFixedPackingGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckedFixedPackingGraph")
            .field("registration", &self.registration())
            .field("inspection", &self.inspection)
            .finish()
    }
}
impl CheckedFixedPackingGraph {
    pub fn verify(
        artifacts: RegisteredPackingArtifactBytes,
        budget: PackingGraphInspectionBudget,
    ) -> Result<Self> {
        validate_budget(budget)?;
        let registration = artifacts.registration();
        let bytes = artifacts.graph_bytes();
        if bytes.is_empty() || bytes.len() > DEVICE_PACKING_MAX_GRAPH_BYTES {
            return Err(PackingGraphBodyError::ArtifactBounds);
        }
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        if u64::try_from(bytes.len()).map_err(|_| PackingGraphBodyError::Overflow)?
            != registration.graph.bytes
            || digest != registration.graph.sha256
            || <[u8; 32]>::from(Sha256::digest(artifacts.manifest_bytes()))
                != registration.manifest.sha256
            || u64::try_from(artifacts.manifest_bytes().len())
                .map_err(|_| PackingGraphBodyError::Overflow)?
                != registration.manifest.bytes
        {
            return Err(PackingGraphBodyError::ByteIdentityMismatch);
        }
        let inspection = inspect_graph(bytes, budget)?;
        Ok(Self {
            artifacts,
            inspection,
        })
    }
    pub fn graph_bytes(&self) -> &[u8] {
        self.artifacts.graph_bytes()
    }
    pub fn manifest_bytes(&self) -> &[u8] {
        self.artifacts.manifest_bytes()
    }
    pub fn registration(&self) -> PackingArtifactRegistration {
        self.artifacts.registration()
    }
    pub fn inspection(&self) -> &PackingGraphBodyInspection {
        &self.inspection
    }
    pub fn verification_scope(&self) -> PackingGraphBodyVerification {
        PackingGraphBodyVerification::FixedWeightFreeGraphBodyChecked
    }
    pub fn native_verification(&self) -> PackingNativeVerification {
        PackingNativeVerification::NotPerformed
    }
    /// Original declarations remain declarations; body inspection does not
    /// measure a native session or replace the artifact owner's resource gate.
    pub fn registered_artifacts(&self) -> &RegisteredPackingArtifactBytes {
        &self.artifacts
    }
    /// Exact retained Vec capacities/header from byte admission, plus this
    /// wrapper's additional inline metadata. Inspection scratch is released
    /// before this object is returned and is reported separately in inspection.
    pub fn known_retained_host_bytes(&self) -> Result<u64> {
        let additional = size_of::<Self>()
            .checked_sub(size_of::<RegisteredPackingArtifactBytes>())
            .ok_or(PackingGraphBodyError::Overflow)?;
        self.artifacts
            .resources()
            .owned_artifact_host_bytes
            .checked_add(u64::try_from(additional).map_err(|_| PackingGraphBodyError::Overflow)?)
            .ok_or(PackingGraphBodyError::Overflow)
    }
    /// Inventories are the fixed specification already matched to the actual
    /// body. They do not accept a caller inventory or attest provider placement.
    pub fn input_descriptor(&self, index: usize) -> Result<PackingGraphTensorDescriptor> {
        let (name, code, shape) = input_spec(index)?;
        Ok(PackingGraphTensorDescriptor {
            name,
            dtype: PackingGraphDataType::from_code(code)?,
            shape,
        })
    }
    pub fn output_descriptor(&self, index: usize) -> Result<PackingGraphTensorDescriptor> {
        let (name, code, shape) = output_spec(index)?;
        Ok(PackingGraphTensorDescriptor {
            name,
            dtype: PackingGraphDataType::from_code(code)?,
            shape,
        })
    }
    pub fn node_descriptor(&self, index: usize) -> Result<PackingGraphNodeDescriptor> {
        let spec = NodeSpec::new(index)?;
        let name = spec.name()?;
        Ok(PackingGraphNodeDescriptor {
            name,
            operation: std::str::from_utf8(spec.operation())
                .map_err(|_| PackingGraphBodyError::FixedGraphMismatch("descriptor operation"))?,
            output: PackingGraphTensorDescriptor {
                name,
                dtype: PackingGraphDataType::from_code(spec.dtype())?,
                shape: spec.shape()?,
            },
            input_count: spec.input_count(),
            attribute_count: spec.attribute_count(),
        })
    }
}

fn validate_budget(budget: PackingGraphInspectionBudget) -> Result<()> {
    if budget.max_inspection_host_bytes == 0
        || budget.max_inspection_host_bytes > PACKING_GRAPH_MAX_INSPECTION_HOST_BYTES
        || budget.max_wire_fields == 0
        || budget.max_wire_fields > PACKING_GRAPH_MAX_WIRE_FIELDS
    {
        return Err(PackingGraphBodyError::InvalidBudget);
    }
    Ok(())
}

struct Work {
    budget: PackingGraphInspectionBudget,
    units: u64,
    host_reserved: u64,
}
impl Work {
    fn new(budget: PackingGraphInspectionBudget) -> Result<Self> {
        validate_budget(budget)?;
        let fixed = FIXED_SCRATCH_RESERVATION
            .checked_add(size_of::<CheckedFixedPackingGraph>() as u64)
            .and_then(|n| n.checked_add(size_of::<GraphParts<'static>>() as u64))
            .ok_or(PackingGraphBodyError::Overflow)?;
        if fixed > budget.max_inspection_host_bytes {
            return Err(PackingGraphBodyError::ResourceExceeded(
                "inspection_host_bytes",
            ));
        }
        Ok(Self {
            budget,
            units: 0,
            host_reserved: fixed,
        })
    }
    fn tick(&mut self) -> Result<()> {
        self.units = self
            .units
            .checked_add(1)
            .ok_or(PackingGraphBodyError::Overflow)?;
        if self.units > self.budget.max_wire_fields {
            return Err(PackingGraphBodyError::ResourceExceeded("wire_fields"));
        }
        Ok(())
    }
    fn reserve<T>(&mut self, count: usize) -> Result<Vec<T>> {
        let requested = count
            .checked_mul(size_of::<T>())
            .ok_or(PackingGraphBodyError::Overflow)?;
        let proposed = self
            .host_reserved
            .checked_add(u64::try_from(requested).map_err(|_| PackingGraphBodyError::Overflow)?)
            .ok_or(PackingGraphBodyError::Overflow)?;
        if proposed > self.budget.max_inspection_host_bytes {
            return Err(PackingGraphBodyError::ResourceExceeded(
                "inspection_host_bytes",
            ));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| PackingGraphBodyError::Allocation)?;
        let capacity_bytes = values
            .capacity()
            .checked_mul(size_of::<T>())
            .ok_or(PackingGraphBodyError::Overflow)?;
        self.host_reserved = self
            .host_reserved
            .checked_add(
                u64::try_from(capacity_bytes).map_err(|_| PackingGraphBodyError::Overflow)?,
            )
            .ok_or(PackingGraphBodyError::Overflow)?;
        if self.host_reserved > self.budget.max_inspection_host_bytes {
            return Err(PackingGraphBodyError::ResourceExceeded(
                "inspection_host_bytes",
            ));
        }
        Ok(values)
    }
}

#[derive(Clone, Copy)]
enum Payload<'a> {
    Integer(u64),
    Bytes(&'a [u8]),
}
#[derive(Clone, Copy)]
struct Field<'a> {
    tag: u32,
    value: Payload<'a>,
}
impl<'a> Field<'a> {
    fn integer(self) -> Result<u64> {
        match self.value {
            Payload::Integer(v) => Ok(v),
            _ => Err(PackingGraphBodyError::MalformedWire("expected varint")),
        }
    }
    fn bytes(self) -> Result<&'a [u8]> {
        match self.value {
            Payload::Bytes(v) => Ok(v),
            _ => Err(PackingGraphBodyError::MalformedWire(
                "expected length-delimited field",
            )),
        }
    }
}
struct Wire<'a> {
    bytes: &'a [u8],
    cursor: usize,
    depth: u8,
}
impl<'a> Wire<'a> {
    fn new(bytes: &'a [u8], depth: u8) -> Result<Self> {
        if depth > MAX_DEPTH {
            return Err(PackingGraphBodyError::ResourceExceeded("message_depth"));
        }
        Ok(Self {
            bytes,
            cursor: 0,
            depth,
        })
    }
    fn next(&mut self, work: &mut Work) -> Result<Option<Field<'a>>> {
        if self.cursor == self.bytes.len() {
            return Ok(None);
        }
        work.tick()?;
        let key = self.varint()?;
        let tag = key >> 3;
        if tag == 0 || tag > 0x1fff_ffff {
            return Err(PackingGraphBodyError::MalformedWire("invalid field number"));
        }
        let value = match key & 7 {
            0 => Payload::Integer(self.varint()?),
            2 => {
                let length =
                    usize::try_from(self.varint()?).map_err(|_| PackingGraphBodyError::Overflow)?;
                let end = self
                    .cursor
                    .checked_add(length)
                    .ok_or(PackingGraphBodyError::Overflow)?;
                let bytes = self
                    .bytes
                    .get(self.cursor..end)
                    .ok_or(PackingGraphBodyError::MalformedWire("truncated field"))?;
                self.cursor = end;
                Payload::Bytes(bytes)
            }
            _ => {
                return Err(PackingGraphBodyError::MalformedWire(
                    "unsupported wire type",
                ))
            }
        };
        Ok(Some(Field {
            tag: tag as u32,
            value,
        }))
    }
    fn varint(&mut self) -> Result<u64> {
        let mut value = 0_u64;
        for shift in 0..10 {
            let byte = *self
                .bytes
                .get(self.cursor)
                .ok_or(PackingGraphBodyError::MalformedWire("truncated varint"))?;
            self.cursor = self
                .cursor
                .checked_add(1)
                .ok_or(PackingGraphBodyError::Overflow)?;
            if shift == 9 && byte > 1 {
                return Err(PackingGraphBodyError::MalformedWire("varint overflow"));
            }
            value |= u64::from(byte & 127) << (7 * shift);
            if byte & 128 == 0 {
                if shift != 0 && byte == 0 {
                    return Err(PackingGraphBodyError::MalformedWire("noncanonical varint"));
                }
                return Ok(value);
            }
        }
        Err(PackingGraphBodyError::MalformedWire("varint overflow"))
    }
    fn child(&self, bytes: &'a [u8]) -> Result<Wire<'a>> {
        Wire::new(
            bytes,
            self.depth
                .checked_add(1)
                .ok_or(PackingGraphBodyError::Overflow)?,
        )
    }
}
fn once<T>(slot: &mut Option<T>, value: T) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(PackingGraphBodyError::MalformedWire(
            "duplicate singular field",
        ));
    }
    Ok(())
}
fn exact(condition: bool, detail: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(PackingGraphBodyError::FixedGraphMismatch(detail))
    }
}
fn unsupported(message: &'static str, tag: u32) -> PackingGraphBodyError {
    PackingGraphBodyError::UnsupportedField { message, tag }
}
fn push_bounded<'a>(values: &mut Vec<&'a [u8]>, value: &'a [u8], limit: usize) -> Result<()> {
    if values.len() == limit {
        return Err(PackingGraphBodyError::FixedGraphMismatch(
            "repeated field count",
        ));
    }
    values.push(value); // Capacity is checked/reserved before any graph fields.
    Ok(())
}

struct GraphParts<'a> {
    nodes: Vec<&'a [u8]>,
    initializers: Vec<&'a [u8]>,
    inputs: Vec<&'a [u8]>,
    outputs: Vec<&'a [u8]>,
    values: Vec<&'a [u8]>,
}
fn inspect_graph(
    bytes: &[u8],
    budget: PackingGraphInspectionBudget,
) -> Result<PackingGraphBodyInspection> {
    if bytes.is_empty() || bytes.len() > DEVICE_PACKING_MAX_GRAPH_BYTES {
        return Err(PackingGraphBodyError::ArtifactBounds);
    }
    let mut work = Work::new(budget)?;
    let mut model = Wire::new(bytes, 1)?;
    let (mut ir, mut producer, mut version, mut graph, mut opset) = (None, None, None, None, None);
    let mut metadata = [false; 3];
    while let Some(field) = model.next(&mut work)? {
        match field.tag {
            1 => once(&mut ir, field.integer()?)?,
            2 => once(&mut producer, field.bytes()?)?,
            3 => once(&mut version, field.bytes()?)?,
            7 => once(&mut graph, field.bytes()?)?,
            8 => once(&mut opset, field.bytes()?)?,
            14 => {
                let (key, value) = string_pair(model.child(field.bytes()?)?, &mut work)?;
                let index = match key {
                    b"artifact_domain" => {
                        exact(
                            value == DEVICE_PACKING_DOMAIN.as_bytes(),
                            "artifact domain metadata",
                        )?;
                        0
                    }
                    b"learned_weights" => {
                        exact(value == b"0", "learned weights metadata")?;
                        1
                    }
                    b"native_connected" => {
                        exact(value == b"false", "native scope metadata")?;
                        2
                    }
                    _ => return Err(PackingGraphBodyError::FixedGraphMismatch("metadata key")),
                };
                exact(!metadata[index], "duplicate metadata key")?;
                metadata[index] = true;
            }
            tag => return Err(unsupported("ModelProto", tag)),
        }
    }
    exact(
        ir == Some(10)
            && producer == Some(b"rz-pals-device-packing".as_slice())
            && version == Some(b"1".as_slice())
            && metadata == [true; 3],
        "model header",
    )?;
    validate_opset(
        model.child(opset.ok_or(PackingGraphBodyError::FixedGraphMismatch("missing opset"))?)?,
        &mut work,
    )?;
    let mut graph_wire =
        model.child(graph.ok_or(PackingGraphBodyError::FixedGraphMismatch("missing graph"))?)?;
    let mut parts = GraphParts {
        nodes: work.reserve(NODES)?,
        initializers: work.reserve(INITIALIZERS)?,
        inputs: work.reserve(INPUTS)?,
        outputs: work.reserve(OUTPUTS)?,
        values: work.reserve(VALUES)?,
    };
    let mut name = None;
    while let Some(field) = graph_wire.next(&mut work)? {
        match field.tag {
            1 => push_bounded(&mut parts.nodes, field.bytes()?, NODES)?,
            2 => once(&mut name, field.bytes()?)?,
            5 => push_bounded(&mut parts.initializers, field.bytes()?, INITIALIZERS)?,
            11 => push_bounded(&mut parts.inputs, field.bytes()?, INPUTS)?,
            12 => push_bounded(&mut parts.outputs, field.bytes()?, OUTPUTS)?,
            13 => push_bounded(&mut parts.values, field.bytes()?, VALUES)?,
            tag => return Err(unsupported("GraphProto", tag)),
        }
    }
    exact(name == Some(DEVICE_PACKING_DOMAIN.as_bytes()), "graph name")?;
    exact(
        parts.nodes.len() == NODES
            && parts.inputs.len() == INPUTS
            && parts.outputs.len() == OUTPUTS
            && parts.values.len() == VALUES
            && parts.initializers.len() == INITIALIZERS,
        "graph counts",
    )?;
    for (index, raw) in parts.initializers.iter().enumerate() {
        validate_initializer(graph_wire.child(raw)?, index, &mut work)?;
    }
    for (index, raw) in parts.inputs.iter().enumerate() {
        let (name, dtype, shape) = input_spec(index)?;
        validate_value(graph_wire.child(raw)?, name, dtype, shape, &mut work)?;
    }
    for (index, raw) in parts.outputs.iter().enumerate() {
        let (name, dtype, shape) = output_spec(index)?;
        validate_value(graph_wire.child(raw)?, name, dtype, shape, &mut work)?;
    }
    let mut value_index = 0;
    for (index, raw) in parts.nodes.iter().enumerate() {
        let spec = NodeSpec::new(index)?;
        validate_node(graph_wire.child(raw)?, spec, &mut work)?;
        if spec.local != 130 && index != NODES - 1 {
            // Only memory_key/value and finite_output are graph outputs.
            let value =
                parts
                    .values
                    .get(value_index)
                    .ok_or(PackingGraphBodyError::FixedGraphMismatch(
                        "missing value_info",
                    ))?;
            validate_value(
                graph_wire.child(value)?,
                spec.name()?,
                spec.dtype(),
                spec.shape()?,
                &mut work,
            )?;
            value_index += 1;
        }
    }
    exact(value_index == VALUES, "value_info order/count")?;
    Ok(PackingGraphBodyInspection {
        graph_sha256: Sha256::digest(bytes).into(),
        graph_bytes: u64::try_from(bytes.len()).map_err(|_| PackingGraphBodyError::Overflow)?,
        wire_fields_and_packed_elements: work.units,
        inspection_host_reserved_bytes: work.host_reserved,
        node_count: NODES,
        input_count: INPUTS,
        output_count: OUTPUTS,
        value_info_count: VALUES,
        integer_control_initializer_count: INITIALIZERS,
    })
}

fn string_pair<'a>(mut wire: Wire<'a>, work: &mut Work) -> Result<(&'a [u8], &'a [u8])> {
    let (mut key, mut value) = (None, None);
    while let Some(field) = wire.next(work)? {
        match field.tag {
            1 => once(&mut key, field.bytes()?)?,
            2 => once(&mut value, field.bytes()?)?,
            tag => return Err(unsupported("StringStringEntryProto", tag)),
        }
    }
    Ok((
        key.ok_or(PackingGraphBodyError::FixedGraphMismatch(
            "metadata key absent",
        ))?,
        value.ok_or(PackingGraphBodyError::FixedGraphMismatch(
            "metadata value absent",
        ))?,
    ))
}
fn validate_opset(mut wire: Wire<'_>, work: &mut Work) -> Result<()> {
    let (mut domain, mut version) = (None, None);
    while let Some(field) = wire.next(work)? {
        match field.tag {
            1 => once(&mut domain, field.bytes()?)?,
            2 => once(&mut version, field.integer()?)?,
            tag => return Err(unsupported("OperatorSetIdProto", tag)),
        }
    }
    exact(
        domain.unwrap_or(b"").is_empty() && version == Some(17),
        "opset",
    )
}
fn integers(
    field: Field<'_>,
    output: &mut [u64],
    count: &mut usize,
    work: &mut Work,
) -> Result<()> {
    let mut append = |value: u64, work: &mut Work| -> Result<()> {
        work.tick()?;
        let slot = output
            .get_mut(*count)
            .ok_or(PackingGraphBodyError::FixedGraphMismatch("numeric count"))?;
        *slot = value;
        *count = count
            .checked_add(1)
            .ok_or(PackingGraphBodyError::Overflow)?;
        Ok(())
    };
    match field.value {
        Payload::Integer(value) => append(value, work),
        Payload::Bytes(bytes) => {
            let mut packed = Wire::new(bytes, 1)?;
            while packed.cursor != bytes.len() {
                append(packed.varint()?, work)?;
            }
            Ok(())
        }
    }
}
fn validate_initializer(mut wire: Wire<'_>, index: usize, work: &mut Work) -> Result<()> {
    let names = [
        b"start_zero".as_slice(),
        b"board_stop",
        b"token_axis",
        b"unit_step",
        b"finite_zero",
    ];
    let values = [0, 66, 2, 1, 0];
    let (mut name, mut dtype) = (None, None);
    let (mut dims, mut data, mut dim_count, mut data_count) = ([0; 1], [0; 1], 0, 0);
    while let Some(field) = wire.next(work)? {
        match field.tag {
            1 => integers(field, &mut dims, &mut dim_count, work)?,
            2 => once(&mut dtype, field.integer()?)?,
            7 => integers(field, &mut data, &mut data_count, work)?,
            8 => once(&mut name, field.bytes()?)?,
            tag => return Err(unsupported("TensorProto", tag)),
        }
    }
    exact(
        name == Some(names[index])
            && dtype == Some(INT64)
            && data_count == 1
            && data[0] == values[index]
            && if index == 4 {
                dim_count == 0
            } else {
                dim_count == 1 && dims[0] == 1
            },
        "control initializer",
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingGraphName {
    bytes: [u8; 64],
    len: usize,
}
type Name = PackingGraphName;
impl PackingGraphName {
    fn from_parts(parts: &[&str]) -> Result<Self> {
        let mut result = Self {
            bytes: [0; 64],
            len: 0,
        };
        for part in parts {
            let end = result
                .len
                .checked_add(part.len())
                .ok_or(PackingGraphBodyError::Overflow)?;
            let destination = result.bytes.get_mut(result.len..end).ok_or(
                PackingGraphBodyError::FixedGraphMismatch("specification name bound"),
            )?;
            destination.copy_from_slice(part.as_bytes());
            result.len = end;
        }
        Ok(result)
    }
    fn indexed(parts: &[&str], index: usize) -> Result<Self> {
        exact(index < 1000, "specification index bound")?;
        let mut result = Self::from_parts(parts)?;
        let end = result
            .len
            .checked_add(3)
            .ok_or(PackingGraphBodyError::Overflow)?;
        let destination = result.bytes.get_mut(result.len..end).ok_or(
            PackingGraphBodyError::FixedGraphMismatch("specification name bound"),
        )?;
        destination.copy_from_slice(&[
            b'0' + (index / 100) as u8,
            b'0' + ((index / 10) % 10) as u8,
            b'0' + (index % 10) as u8,
        ]);
        result.len = end;
        Ok(result)
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
    pub fn as_str(&self) -> Result<&str> {
        std::str::from_utf8(self.as_bytes())
            .map_err(|_| PackingGraphBodyError::FixedGraphMismatch("descriptor name"))
    }
    fn bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackingGraphDimension {
    Number(u64),
    Symbol(PackingGraphName),
}
type Dimension = PackingGraphDimension;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingGraphShape {
    dims: [Dimension; 4],
    len: usize,
}
type Shape = PackingGraphShape;
impl PackingGraphShape {
    pub fn dimensions(&self) -> &[PackingGraphDimension] {
        &self.dims[..self.len]
    }
    fn scalar() -> Self {
        Self {
            dims: [Dimension::Number(0); 4],
            len: 0,
        }
    }
    fn vector() -> Self {
        Self {
            dims: [Dimension::Number(1); 4],
            len: 1,
        }
    }
    fn kv(tokens: Dimension) -> Self {
        Self {
            dims: [
                Dimension::Number(1),
                Dimension::Number(2),
                tokens,
                Dimension::Number(64),
            ],
            len: 4,
        }
    }
}
fn input_spec(index: usize) -> Result<(Name, u64, Shape)> {
    match index {
        0 | 1 => Ok((
            Name::from_parts(&[if index == 0 {
                "board_key"
            } else {
                "board_value"
            }])?,
            FLOAT,
            Shape::kv(Dimension::Symbol(Name::from_parts(&[
                "base_source_tokens",
            ])?)),
        )),
        386 => Ok((Name::from_parts(&["record_stop"])?, INT64, Shape::vector())),
        2..=385 => {
            let record = (index - 2) / 3;
            let role = (index - 2) % 3;
            let name = Name::indexed(
                &[match role {
                    0 => "record_key_",
                    1 => "record_value_",
                    _ => "record_offset_",
                }],
                record,
            )?;
            if role == 2 {
                Ok((name, INT64, Shape::vector()))
            } else {
                Ok((
                    name,
                    FLOAT,
                    Shape::kv(Dimension::Symbol(Name::indexed(
                        &["source_tokens_"],
                        record,
                    )?)),
                ))
            }
        }
        _ => Err(PackingGraphBodyError::FixedGraphMismatch("input index")),
    }
}
fn output_spec(index: usize) -> Result<(Name, u64, Shape)> {
    match index {
        0 | 1 => Ok((
            Name::from_parts(&[if index == 0 {
                "memory_key"
            } else {
                "memory_value"
            }])?,
            FLOAT,
            Shape::kv(Dimension::Symbol(Name::from_parts(&["memory_tokens"])?)),
        )),
        2 => Ok((Name::from_parts(&["finite_output"])?, BOOL, Shape::scalar())),
        _ => Err(PackingGraphBodyError::FixedGraphMismatch("output index")),
    }
}
fn validate_value(
    mut wire: Wire<'_>,
    expected_name: Name,
    dtype: u64,
    shape: Shape,
    work: &mut Work,
) -> Result<()> {
    let (mut name, mut ty) = (None, None);
    while let Some(field) = wire.next(work)? {
        match field.tag {
            1 => once(&mut name, field.bytes()?)?,
            2 => once(&mut ty, field.bytes()?)?,
            tag => return Err(unsupported("ValueInfoProto", tag)),
        }
    }
    exact(name == Some(expected_name.bytes()), "value name/order")?;
    let mut type_wire = wire.child(ty.ok_or(PackingGraphBodyError::FixedGraphMismatch(
        "missing value type",
    ))?)?;
    let mut tensor = None;
    while let Some(field) = type_wire.next(work)? {
        match field.tag {
            1 => once(&mut tensor, field.bytes()?)?,
            tag => return Err(unsupported("TypeProto", tag)),
        }
    }
    let mut tensor_wire = type_wire.child(tensor.ok_or(
        PackingGraphBodyError::FixedGraphMismatch("missing tensor type"),
    )?)?;
    let (mut actual_dtype, mut actual_shape) = (None, None);
    while let Some(field) = tensor_wire.next(work)? {
        match field.tag {
            1 => once(&mut actual_dtype, field.integer()?)?,
            2 => once(&mut actual_shape, field.bytes()?)?,
            tag => return Err(unsupported("TypeProto.Tensor", tag)),
        }
    }
    exact(actual_dtype == Some(dtype), "value dtype")?;
    let mut shape_wire = tensor_wire.child(actual_shape.ok_or(
        PackingGraphBodyError::FixedGraphMismatch("missing tensor shape"),
    )?)?;
    let mut index = 0;
    while let Some(field) = shape_wire.next(work)? {
        if field.tag != 1 {
            return Err(unsupported("TensorShapeProto", field.tag));
        }
        exact(index < shape.len, "tensor rank")?;
        let mut dimension = shape_wire.child(field.bytes()?)?;
        let (mut number, mut symbol) = (None, None);
        while let Some(field) = dimension.next(work)? {
            match field.tag {
                1 => once(&mut number, field.integer()?)?,
                2 => once(&mut symbol, field.bytes()?)?,
                tag => return Err(unsupported("TensorShapeProto.Dimension", tag)),
            }
        }
        exact(
            !(number.is_some() && symbol.is_some()),
            "dimension oneof conflict",
        )?;
        match shape.dims[index] {
            Dimension::Number(n) => {
                exact(number == Some(n) && symbol.is_none(), "fixed dimension")?
            }
            Dimension::Symbol(s) => exact(
                symbol == Some(s.bytes()) && number.is_none(),
                "symbolic dimension",
            )?,
        }
        index += 1;
    }
    exact(index == shape.len, "tensor rank")
}

#[derive(Clone, Copy)]
struct NodeSpec {
    kind: &'static str,
    local: usize,
    final_node: bool,
}
impl NodeSpec {
    fn new(index: usize) -> Result<Self> {
        exact(index < NODES, "node index")?;
        Ok(Self {
            kind: if index < 137 { "key" } else { "value" },
            local: index % 137,
            final_node: index == 274,
        })
    }
    fn name(self) -> Result<Name> {
        if self.final_node {
            return Name::from_parts(&["finite_output"]);
        }
        let stem = match self.local {
            0 => "board_slice_",
            1..=128 => return Name::indexed(&["gather_", self.kind, "_"], self.local - 1),
            129 => "all_ports_",
            130 => "memory_",
            131 => "nan_",
            132 => "inf_",
            133 => "bad_",
            134 => "bad_int_",
            135 => "bad_max_",
            136 => "finite_",
            _ => return Err(PackingGraphBodyError::FixedGraphMismatch("node index")),
        };
        Name::from_parts(&[stem, self.kind])
    }
    fn operation(self) -> &'static [u8] {
        if self.final_node {
            return b"And";
        }
        match self.local {
            0 | 130 => b"Slice",
            1..=128 => b"Gather",
            129 => b"Concat",
            131 => b"IsNaN",
            132 => b"IsInf",
            133 => b"Or",
            134 => b"Cast",
            135 => b"ReduceMax",
            _ => b"Equal",
        }
    }
    fn input_count(self) -> usize {
        if self.final_node {
            return 2;
        }
        match self.local {
            0 | 130 => 5,
            1..=128 | 133 | 136 => 2,
            129 => 129,
            _ => 1,
        }
    }
    fn input(self, slot: usize) -> Result<Name> {
        exact(slot < self.input_count(), "node input count")?;
        if self.final_node {
            return Name::from_parts(&[if slot == 0 {
                "finite_key"
            } else {
                "finite_value"
            }]);
        }
        match self.local {
            0 | 130 => match slot {
                0 => Name::from_parts(&[
                    if self.local == 0 {
                        "board_"
                    } else {
                        "all_ports_"
                    },
                    self.kind,
                ]),
                1 => Name::from_parts(&["start_zero"]),
                2 => Name::from_parts(&[if self.local == 0 {
                    "board_stop"
                } else {
                    "record_stop"
                }]),
                3 => Name::from_parts(&["token_axis"]),
                _ => Name::from_parts(&["unit_step"]),
            },
            1..=128 => {
                if slot == 0 {
                    Name::indexed(&["record_", self.kind, "_"], self.local - 1)
                } else {
                    Name::indexed(&["record_offset_"], self.local - 1)
                }
            }
            129 => {
                if slot == 0 {
                    Name::from_parts(&["board_slice_", self.kind])
                } else {
                    Name::indexed(&["gather_", self.kind, "_"], slot - 1)
                }
            }
            131 | 132 => Name::from_parts(&["memory_", self.kind]),
            133 => Name::from_parts(&[if slot == 0 { "nan_" } else { "inf_" }, self.kind]),
            134 => Name::from_parts(&["bad_", self.kind]),
            135 => Name::from_parts(&["bad_int_", self.kind]),
            136 => {
                if slot == 0 {
                    Name::from_parts(&["bad_max_", self.kind])
                } else {
                    Name::from_parts(&["finite_zero"])
                }
            }
            _ => Err(PackingGraphBodyError::FixedGraphMismatch(
                "node input index",
            )),
        }
    }
    fn dtype(self) -> u64 {
        if self.final_node {
            return BOOL;
        }
        match self.local {
            0..=130 => FLOAT,
            134 | 135 => INT64,
            _ => BOOL,
        }
    }
    fn shape(self) -> Result<Shape> {
        if self.final_node || self.local >= 135 {
            return Ok(Shape::scalar());
        }
        Ok(Shape::kv(match self.local {
            0 => Dimension::Number(66),
            1..=128 => Dimension::Number(1),
            129 => Dimension::Number(194),
            _ => Dimension::Symbol(Name::from_parts(&["memory_tokens"])?),
        }))
    }
    fn attribute_count(self) -> usize {
        if self.final_node {
            0
        } else {
            match self.local {
                1..=129 | 134 => 1,
                135 => 2,
                _ => 0,
            }
        }
    }
}
fn validate_node(mut wire: Wire<'_>, spec: NodeSpec, work: &mut Work) -> Result<()> {
    let (mut name, mut operation, mut domain) = (None, None, None);
    let (mut inputs, mut outputs, mut attributes) = (0, 0, 0);
    while let Some(field) = wire.next(work)? {
        match field.tag {
            1 => {
                let expected = spec.input(inputs)?;
                exact(
                    field.bytes()? == expected.bytes(),
                    "node ordered input/edge",
                )?;
                inputs += 1;
            }
            2 => {
                exact(
                    outputs == 0 && field.bytes()? == spec.name()?.bytes(),
                    "node output",
                )?;
                outputs += 1;
            }
            3 => once(&mut name, field.bytes()?)?,
            4 => once(&mut operation, field.bytes()?)?,
            5 => {
                exact(attributes < spec.attribute_count(), "node attribute count")?;
                validate_attribute(wire.child(field.bytes()?)?, spec, attributes, work)?;
                attributes += 1;
            }
            7 => once(&mut domain, field.bytes()?)?,
            tag => return Err(unsupported("NodeProto", tag)),
        }
    }
    exact(
        name == Some(spec.name()?.bytes())
            && operation == Some(spec.operation())
            && domain.unwrap_or(b"").is_empty()
            && inputs == spec.input_count()
            && outputs == 1
            && attributes == spec.attribute_count(),
        "node header/counts",
    )
}
fn validate_attribute(
    mut wire: Wire<'_>,
    spec: NodeSpec,
    index: usize,
    work: &mut Work,
) -> Result<()> {
    let (mut name, mut ty, mut integer) = (None, None, None);
    let (mut ints, mut count) = ([0; 4], 0);
    let mut has_ints_field = false;
    while let Some(field) = wire.next(work)? {
        match field.tag {
            1 => once(&mut name, field.bytes()?)?,
            3 => once(&mut integer, field.integer()?)?,
            8 => {
                has_ints_field = true;
                integers(field, &mut ints, &mut count, work)?;
            }
            20 => once(&mut ty, field.integer()?)?,
            tag => return Err(unsupported("AttributeProto", tag)),
        }
    }
    if spec.local == 135 && index == 0 {
        exact(
            name == Some(b"axes".as_slice())
                && ty == Some(7)
                && integer.is_none()
                && count == 4
                && ints == [0, 1, 2, 3],
            "ReduceMax axes",
        )
    } else {
        let (expected_name, value) = match spec.local {
            1..=129 => (b"axis".as_slice(), 2),
            134 => (b"to".as_slice(), INT64),
            135 => (b"keepdims".as_slice(), 0),
            _ => {
                return Err(PackingGraphBodyError::FixedGraphMismatch(
                    "unexpected attribute",
                ))
            }
        };
        // Proto3 can omit a scalar zero. INT type plus absent i denotes zero;
        // no other value field is admitted and duplicate singulars still fail.
        exact(
            name == Some(expected_name)
                && ty == Some(2)
                && integer.unwrap_or(0) == value
                && !has_ints_field
                && count == 0,
            "integer attribute",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::device_packing_admission::{
        DevicePackingResourceDeclaration, RegisteredPackingBytePin, DEVICE_PACKING_GRAPH_FILE,
        DEVICE_PACKING_SCHEMA,
    };
    use super::*;
    use serde_json::{json, Value};

    // Independent synthetic encoding of the Python generator's recipe using
    // the official schema tags. No protobuf library, Python import, ONNX checker,
    // ORT Session, model, provider, or actual generated artifact is used here.
    fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 127) as u8;
            value >>= 7;
            out.push(byte | if value == 0 { 0 } else { 128 });
            if value == 0 {
                return out;
            }
        }
    }
    fn integer(tag: u64, value: u64) -> Vec<u8> {
        let mut out = varint(tag << 3);
        out.extend(varint(value));
        out
    }
    fn bytes(tag: u64, value: &[u8]) -> Vec<u8> {
        let mut out = varint((tag << 3) | 2);
        out.extend(varint(value.len() as u64));
        out.extend(value);
        out
    }
    fn text(tag: u64, value: &str) -> Vec<u8> {
        bytes(tag, value.as_bytes())
    }
    fn packed(tag: u64, values: &[u64]) -> Vec<u8> {
        bytes(
            tag,
            &values.iter().flat_map(|n| varint(*n)).collect::<Vec<_>>(),
        )
    }
    fn attribute(name: &str, values: &[u64], is_list: bool) -> Vec<u8> {
        let mut out = text(1, name);
        out.extend(if is_list {
            packed(8, values)
        } else {
            integer(3, values[0])
        });
        out.extend(integer(20, if is_list { 7 } else { 2 }));
        out
    }
    fn initializer(name: &str, dims: &[u64], value: u64) -> Vec<u8> {
        let mut out = packed(1, dims);
        out.extend(integer(2, 7));
        out.extend(packed(7, &[value]));
        out.extend(text(8, name));
        out
    }
    fn value_info(name: &str, dtype: u64, dimensions: &[Value]) -> Vec<u8> {
        let mut shape = Vec::new();
        for dimension in dimensions {
            let raw = if let Some(n) = dimension.as_u64() {
                integer(1, n)
            } else {
                text(2, dimension.as_str().unwrap())
            };
            shape.extend(bytes(1, &raw));
        }
        let mut tensor = integer(1, dtype);
        tensor.extend(bytes(2, &shape));
        let mut out = text(1, name);
        out.extend(bytes(2, &bytes(1, &tensor)));
        out
    }
    struct FixtureNode {
        name: String,
        op: String,
        inputs: Vec<String>,
        attrs: Vec<Vec<u8>>,
        extra: Vec<u8>,
    }
    impl FixtureNode {
        fn encode(&self) -> Vec<u8> {
            let mut out = Vec::new();
            for name in &self.inputs {
                out.extend(text(1, name));
            }
            out.extend(text(2, &self.name));
            out.extend(text(3, &self.name));
            out.extend(text(4, &self.op));
            for attr in &self.attrs {
                out.extend(bytes(5, attr));
            }
            out.extend(&self.extra);
            out
        }
    }
    struct Fixture {
        nodes: Vec<FixtureNode>,
        initializers: Vec<Vec<u8>>,
        inputs: Vec<Vec<u8>>,
        outputs: Vec<Vec<u8>>,
        values: Vec<Vec<u8>>,
        graph_extra: Vec<u8>,
        model_extra: Vec<u8>,
    }
    impl Fixture {
        fn new() -> Self {
            let mut fixture = Self {
                nodes: Vec::new(),
                initializers: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                values: Vec::new(),
                graph_extra: Vec::new(),
                model_extra: Vec::new(),
            };
            fixture.initializers = vec![
                initializer("start_zero", &[1], 0),
                initializer("board_stop", &[1], 66),
                initializer("token_axis", &[1], 2),
                initializer("unit_step", &[1], 1),
                initializer("finite_zero", &[], 0),
            ];
            let base = vec![json!(1), json!(2), json!("base_source_tokens"), json!(64)];
            fixture.inputs.push(value_info("board_key", 1, &base));
            fixture.inputs.push(value_info("board_value", 1, &base));
            for i in 0..128 {
                let shape = vec![
                    json!(1),
                    json!(2),
                    json!(format!("source_tokens_{i:03}")),
                    json!(64),
                ];
                fixture
                    .inputs
                    .push(value_info(&format!("record_key_{i:03}"), 1, &shape));
                fixture
                    .inputs
                    .push(value_info(&format!("record_value_{i:03}"), 1, &shape));
                fixture
                    .inputs
                    .push(value_info(&format!("record_offset_{i:03}"), 7, &[json!(1)]));
            }
            fixture
                .inputs
                .push(value_info("record_stop", 7, &[json!(1)]));
            let final_shape = vec![json!(1), json!(2), json!("memory_tokens"), json!(64)];
            fixture.outputs = vec![
                value_info("memory_key", 1, &final_shape),
                value_info("memory_value", 1, &final_shape),
                value_info("finite_output", 9, &[]),
            ];
            for kind in ["key", "value"] {
                let board = format!("board_slice_{kind}");
                fixture.node(
                    "Slice",
                    vec![
                        format!("board_{kind}"),
                        "start_zero".into(),
                        "board_stop".into(),
                        "token_axis".into(),
                        "unit_step".into(),
                    ],
                    &board,
                    1,
                    vec![json!(1), json!(2), json!(66), json!(64)],
                    vec![],
                );
                let mut joined_inputs = vec![board];
                for i in 0..128 {
                    let gathered = format!("gather_{kind}_{i:03}");
                    fixture.node(
                        "Gather",
                        vec![
                            format!("record_{kind}_{i:03}"),
                            format!("record_offset_{i:03}"),
                        ],
                        &gathered,
                        1,
                        vec![json!(1), json!(2), json!(1), json!(64)],
                        vec![attribute("axis", &[2], false)],
                    );
                    joined_inputs.push(gathered);
                }
                fixture.node(
                    "Concat",
                    joined_inputs,
                    &format!("all_ports_{kind}"),
                    1,
                    vec![json!(1), json!(2), json!(194), json!(64)],
                    vec![attribute("axis", &[2], false)],
                );
                fixture.node(
                    "Slice",
                    vec![
                        format!("all_ports_{kind}"),
                        "start_zero".into(),
                        "record_stop".into(),
                        "token_axis".into(),
                        "unit_step".into(),
                    ],
                    &format!("memory_{kind}"),
                    1,
                    final_shape.clone(),
                    vec![],
                );
                fixture.node(
                    "IsNaN",
                    vec![format!("memory_{kind}")],
                    &format!("nan_{kind}"),
                    9,
                    final_shape.clone(),
                    vec![],
                );
                fixture.node(
                    "IsInf",
                    vec![format!("memory_{kind}")],
                    &format!("inf_{kind}"),
                    9,
                    final_shape.clone(),
                    vec![],
                );
                fixture.node(
                    "Or",
                    vec![format!("nan_{kind}"), format!("inf_{kind}")],
                    &format!("bad_{kind}"),
                    9,
                    final_shape.clone(),
                    vec![],
                );
                fixture.node(
                    "Cast",
                    vec![format!("bad_{kind}")],
                    &format!("bad_int_{kind}"),
                    7,
                    final_shape.clone(),
                    vec![attribute("to", &[7], false)],
                );
                fixture.node(
                    "ReduceMax",
                    vec![format!("bad_int_{kind}")],
                    &format!("bad_max_{kind}"),
                    7,
                    vec![],
                    vec![
                        attribute("axes", &[0, 1, 2, 3], true),
                        attribute("keepdims", &[0], false),
                    ],
                );
                fixture.node(
                    "Equal",
                    vec![format!("bad_max_{kind}"), "finite_zero".into()],
                    &format!("finite_{kind}"),
                    9,
                    vec![],
                    vec![],
                );
            }
            fixture.node(
                "And",
                vec!["finite_key".into(), "finite_value".into()],
                "finite_output",
                9,
                vec![],
                vec![],
            );
            fixture
        }
        fn node(
            &mut self,
            op: &str,
            inputs: Vec<String>,
            name: &str,
            dtype: u64,
            shape: Vec<Value>,
            attrs: Vec<Vec<u8>>,
        ) {
            self.nodes.push(FixtureNode {
                name: name.into(),
                op: op.into(),
                inputs,
                attrs,
                extra: vec![],
            });
            if !["memory_key", "memory_value", "finite_output"].contains(&name) {
                self.values.push(value_info(name, dtype, &shape));
            }
        }
        fn encode(&self) -> Vec<u8> {
            let mut graph = Vec::new();
            for node in &self.nodes {
                graph.extend(bytes(1, &node.encode()));
            }
            graph.extend(text(2, "rovezero.pals.weight-free-public-routing"));
            for init in &self.initializers {
                graph.extend(bytes(5, init));
            }
            for value in &self.inputs {
                graph.extend(bytes(11, value));
            }
            for value in &self.outputs {
                graph.extend(bytes(12, value));
            }
            for value in &self.values {
                graph.extend(bytes(13, value));
            }
            graph.extend(&self.graph_extra);
            let mut model = integer(1, 10);
            model.extend(text(2, "rz-pals-device-packing"));
            model.extend(text(3, "1"));
            model.extend(bytes(7, &graph));
            model.extend(bytes(8, &integer(2, 17)));
            for (key, value) in [
                (
                    "artifact_domain",
                    "rovezero.pals.weight-free-public-routing",
                ),
                ("learned_weights", "0"),
                ("native_connected", "false"),
            ] {
                let mut property = text(1, key);
                property.extend(text(2, value));
                model.extend(bytes(14, &property));
            }
            model.extend(&self.model_extra);
            model
        }
    }
    fn budget() -> PackingGraphInspectionBudget {
        PackingGraphInspectionBudget {
            max_inspection_host_bytes: 64 * 1024,
            max_wire_fields: PACKING_GRAPH_MAX_WIRE_FIELDS,
        }
    }
    fn reject(fixture: &Fixture) {
        assert!(inspect_graph(&fixture.encode(), budget()).is_err());
    }
    fn pin(value: &[u8]) -> RegisteredPackingBytePin {
        RegisteredPackingBytePin {
            bytes: value.len() as u64,
            sha256: Sha256::digest(value).into(),
        }
    }
    fn hex(value: [u8; 32]) -> String {
        value.iter().map(|n| format!("{n:02x}")).collect()
    }
    // Exact declaration fixture is separate from the body recipe. Its pins are
    // caller-supplied synthetic registration, never native producer authority.
    fn manifest(graph: RegisteredPackingBytePin) -> Vec<u8> {
        let tensor = |name: String, dtype: &str, shape: Value| json!({"name":name,"dtype":dtype,"shape":shape});
        let mut inputs = vec![
            tensor(
                "board_key".into(),
                "FLOAT",
                json!([1, 2, "base_source_tokens", 64]),
            ),
            tensor(
                "board_value".into(),
                "FLOAT",
                json!([1, 2, "base_source_tokens", 64]),
            ),
        ];
        for i in 0..128 {
            let shape = json!([1, 2, format!("source_tokens_{i:03}"), 64]);
            inputs.push(tensor(format!("record_key_{i:03}"), "FLOAT", shape.clone()));
            inputs.push(tensor(format!("record_value_{i:03}"), "FLOAT", shape));
            inputs.push(tensor(format!("record_offset_{i:03}"), "INT64", json!([1])));
        }
        inputs.push(tensor("record_stop".into(), "INT64", json!([1])));
        let outputs = vec![
            tensor(
                "memory_key".into(),
                "FLOAT",
                json!([1, 2, "memory_tokens", 64]),
            ),
            tensor(
                "memory_value".into(),
                "FLOAT",
                json!([1, 2, "memory_tokens", 64]),
            ),
            tensor("finite_output".into(), "BOOL", json!([])),
        ];
        let mut ledger = Vec::new();
        let mut add = |name: String, dtype: &str, shape: Vec<u64>, width: u64| {
            let payload = shape.iter().fold(width, |n, d| n * d);
            ledger.push(json!({"node":name,"output":name,"dtype":dtype,"maximum_shape":shape,"maximum_payload_bytes":payload}));
        };
        for kind in ["key", "value"] {
            add(
                format!("board_slice_{kind}"),
                "FLOAT",
                vec![1, 2, 66, 64],
                4,
            );
            for i in 0..128 {
                add(
                    format!("gather_{kind}_{i:03}"),
                    "FLOAT",
                    vec![1, 2, 1, 64],
                    4,
                );
            }
            for stem in ["all_ports", "memory"] {
                add(format!("{stem}_{kind}"), "FLOAT", vec![1, 2, 194, 64], 4);
            }
            for stem in ["nan", "inf", "bad"] {
                add(format!("{stem}_{kind}"), "BOOL", vec![1, 2, 194, 64], 1);
            }
            add(format!("bad_int_{kind}"), "INT64", vec![1, 2, 194, 64], 8);
            add(format!("bad_max_{kind}"), "INT64", vec![], 8);
            add(format!("finite_{kind}"), "BOOL", vec![], 1);
        }
        add("finite_output".into(), "BOOL", vec![], 1);
        serde_json::to_vec(&json!({
            "schema":DEVICE_PACKING_SCHEMA,"artifact_domain":DEVICE_PACKING_DOMAIN,"layout_revision":1,
            "graph":{"file":DEVICE_PACKING_GRAPH_FILE,"bytes":graph.bytes,"sha256":hex(graph.sha256),
                "ir_version":10,"opset":17,"inputs":inputs,"outputs":outputs},
            "learned_weights":{"count":0,"bytes":0},"integer_control_initializers":{"count":5,"payload_bytes":40},
            "bounds":{"batch":1,"precision":"FP32","board_tokens":66,"kv_heads":2,"head_dimension":64,
                "record_capacity":128,"source_tokens_minimum":67,"source_tokens_maximum":194,"record_count_minimum":0,
                "record_count_maximum":128,"record_offset_minimum":66,"record_offset_maximum":193,"record_stop":"66 + max(record_count, 1)"},
            "routing":{"base":"exactly_one_actual_matching_board_base","records":"ordered_occurrences_duplicates_preserved",
                "inactive_ports":"alias_matching_base_token_66","zero_records":"actual_zero_feature_projection_at_token_66_with_false_mask",
                "mask":"CPU_owned_current_mask_first_66_true_zero_pad_false","finite_output":"joined_visible_key_and_value_only"},
            "intermediate_arithmetic":{"meaning":"sum_of_independent_maximum_node_output_payloads_not_peak",
                "node_output_count":275,"node_outputs":ledger,"maximum_sum_payload_bytes":1_142_291,
                "allocator_workspace_and_session_bytes":"unknown","physical_stage_completion":"not_implemented"},
            "scope":{"artifact_and_cpu_numeric_only":true,"native_connected":false,"cuda_resident_owner":"not_implemented",
                "cuda_physical_completion_and_quarantine":"not_implemented","cuda_validation":"not_run_user_deferred",
                "private_warm":"outside_this_artifact","cpu_projection_origin":"caller_identity_and_actual_feature_bits_not_native_producer_attestation",
                "cpu_owner_accounting":"unique_owned_numpy_payloads_only_not_peak","device_runtime_bytes":"unknown"}
        })).unwrap()
    }
    fn admitted(graph: &[u8]) -> RegisteredPackingArtifactBytes {
        let graph_pin = pin(graph);
        let raw_manifest = manifest(graph_pin);
        RegisteredPackingArtifactBytes::admit(
            &raw_manifest,
            graph,
            PackingArtifactRegistration {
                manifest: pin(&raw_manifest),
                graph: graph_pin,
            },
            DevicePackingResourceDeclaration {
                packing_session_bytes: Some(1),
                additional_owner_metadata_host_bytes: Some(1),
                max_owned_artifact_host_bytes: 4 * 1024 * 1024,
                max_declared_host_bytes: 4 * 1024 * 1024,
                max_declared_device_bytes: 4 * 1024 * 1024,
            },
        )
        .unwrap()
    }

    #[test]
    fn synthetic_full_body_retains_registered_pair_without_native_authority() {
        let raw = Fixture::new().encode();
        let artifact = admitted(&raw);
        let retained = artifact.resources().owned_artifact_host_bytes;
        let checked = CheckedFixedPackingGraph::verify(artifact, budget()).unwrap();
        assert_eq!(checked.graph_bytes(), raw);
        assert_eq!(checked.registration().graph, pin(&raw));
        assert_eq!(checked.inspection().node_count, 275);
        assert_eq!(checked.inspection().value_info_count, 272);
        assert_eq!(
            checked.native_verification(),
            PackingNativeVerification::NotPerformed
        );
        assert!(checked.known_retained_host_bytes().unwrap() > retained);
        assert!(
            checked.inspection().inspection_host_reserved_bytes
                <= budget().max_inspection_host_bytes
        );
        assert_eq!(
            checked.input_descriptor(2).unwrap().name.as_str().unwrap(),
            "record_key_000"
        );
        assert_eq!(
            checked
                .input_descriptor(385)
                .unwrap()
                .name
                .as_str()
                .unwrap(),
            "record_offset_127"
        );
        let dims = checked.output_descriptor(0).unwrap().shape;
        assert!(matches!(
            dims.dimensions()[2],
            PackingGraphDimension::Symbol(_)
        ));
        assert_eq!(checked.node_descriptor(135).unwrap().operation, "ReduceMax");
        assert_eq!(checked.node_descriptor(274).unwrap().operation, "And");
        assert!(checked.input_descriptor(387).is_err());
        assert!(checked.output_descriptor(3).is_err());
        assert!(checked.node_descriptor(275).is_err());
    }
    #[test]
    fn repinned_foreign_body_cannot_turn_declarations_into_proof() {
        let mut fixture = Fixture::new();
        fixture.nodes[274].op = "Constant".into();
        let raw = fixture.encode();
        // Both original byte pins and the exact manifest are updated; ordinary
        // byte admission deliberately succeeds, concrete body inspection fails.
        assert!(CheckedFixedPackingGraph::verify(admitted(&raw), budget()).is_err());
    }
    #[test]
    fn ordered_ports_nodes_and_record_edges_are_exact() {
        let mut fixture = Fixture::new();
        fixture.inputs.swap(2, 3);
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes.swap(1, 2);
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes[1].inputs[1] = "record_offset_001".into();
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes[129].inputs.swap(1, 2);
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes[130].inputs[2] = "board_stop".into();
        reject(&fixture);
    }
    #[test]
    fn finite_gate_cannot_be_constant_unused_or_rewired() {
        let mut fixture = Fixture::new();
        fixture.nodes[274].inputs[1] = "finite_key".into();
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes[135].inputs[0] = "record_offset_000".into();
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes.push(FixtureNode {
            name: "unused".into(),
            op: "Identity".into(),
            inputs: vec!["finite_key".into()],
            attrs: vec![],
            extra: vec![],
        });
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.outputs[2] = value_info("finite_key", 9, &[]);
        reject(&fixture);
    }
    #[test]
    fn all_control_values_dtypes_dims_and_payload_forms_are_closed() {
        for index in 0..5 {
            let mut fixture = Fixture::new();
            let names = [
                "start_zero",
                "board_stop",
                "token_axis",
                "unit_step",
                "finite_zero",
            ];
            fixture.initializers[index] =
                initializer(names[index], if index == 4 { &[] } else { &[1] }, 99);
            reject(&fixture);
        }
        let mut fixture = Fixture::new();
        let mut wrong = packed(1, &[1]);
        wrong.extend(integer(2, 1));
        wrong.extend(bytes(4, &[0, 0, 0, 0]));
        wrong.extend(text(8, "start_zero"));
        fixture.initializers[0] = wrong;
        reject(&fixture);
        for tag in [3, 9, 13, 14, 16] {
            let mut fixture = Fixture::new();
            fixture.initializers[0].extend(bytes(tag, b""));
            reject(&fixture);
        }
        let mut fixture = Fixture::new();
        fixture.initializers[4] = initializer("finite_zero", &[1], 0);
        reject(&fixture);
    }
    #[test]
    fn attribute_axis_cast_reduce_and_discriminator_cannot_drift() {
        for (node, attr, replacement) in [
            (1, 0, attribute("axis", &[1], false)),
            (134, 0, attribute("to", &[1], false)),
            (135, 0, attribute("axes", &[0, 1, 2], true)),
            (135, 1, attribute("keepdims", &[1], false)),
        ] {
            let mut fixture = Fixture::new();
            fixture.nodes[node].attrs[attr] = replacement;
            reject(&fixture);
        }
        let mut fixture = Fixture::new();
        fixture.nodes[1].attrs[0].extend(integer(20, 2));
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes[1].attrs[0].extend(packed(8, &[2]));
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes[135].attrs[1].extend(packed(8, &[]));
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes[132]
            .attrs
            .push(attribute("detect_negative", &[0], false));
        reject(&fixture);
    }
    #[test]
    fn shape_rank_dtype_symbol_and_intermediate_order_are_exact() {
        let mut fixture = Fixture::new();
        fixture.inputs[0] =
            value_info("board_key", 1, &[json!(1), json!(2), json!(194), json!(64)]);
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.inputs[2] = value_info(
            "record_key_000",
            1,
            &[json!(1), json!(2), json!("source_tokens_001"), json!(64)],
        );
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.outputs[2] = value_info("finite_output", 1, &[]);
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.values.swap(0, 1);
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.values.pop();
        reject(&fixture);
    }
    #[test]
    fn subgraphs_functions_training_sparse_and_unknown_fields_fail_closed() {
        for tag in [4, 20, 25, 100] {
            let mut fixture = Fixture::new();
            fixture.model_extra = bytes(tag, b"");
            reject(&fixture);
        }
        for tag in [10, 14, 15, 16, 100] {
            let mut fixture = Fixture::new();
            fixture.graph_extra = bytes(tag, b"");
            reject(&fixture);
        }
        let mut fixture = Fixture::new();
        fixture.nodes[1].attrs[0].extend(bytes(6, b""));
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes[1].extra = text(7, "vendor");
        reject(&fixture);
    }
    #[test]
    fn singular_duplicates_and_shape_oneof_conflicts_are_rejected() {
        let mut fixture = Fixture::new();
        fixture.model_extra = integer(1, 10);
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.nodes[0].extra = text(3, "board_slice_key");
        reject(&fixture);
        let mut fixture = Fixture::new();
        fixture.initializers[0].extend(integer(2, 7));
        reject(&fixture);
        let mut conflict = integer(1, 1);
        conflict.extend(text(2, "batch"));
        let mut tensor = integer(1, 1);
        tensor.extend(bytes(2, &bytes(1, &conflict)));
        let mut value = text(1, "board_key");
        value.extend(bytes(2, &bytes(1, &tensor)));
        let mut fixture = Fixture::new();
        fixture.inputs[0] = value;
        reject(&fixture);
    }
    #[test]
    fn legitimate_packed_unpacked_and_proto3_zero_forms_preserve_body_meaning() {
        let mut fixture = Fixture::new();
        let mut init = integer(1, 1);
        init.extend(integer(2, 7));
        init.extend(integer(7, 0));
        init.extend(text(8, "start_zero"));
        fixture.initializers[0] = init;
        let mut keepdims = text(1, "keepdims");
        keepdims.extend(integer(20, 2));
        fixture.nodes[135].attrs[1] = keepdims;
        let mut axes = text(1, "axes");
        for n in [0, 1, 2, 3] {
            axes.extend(integer(8, n));
        }
        axes.extend(integer(20, 7));
        fixture.nodes[135].attrs[0] = axes;
        assert!(inspect_graph(&fixture.encode(), budget()).is_ok());
    }
    #[test]
    fn malformed_wire_depth_and_finite_budget_failures_are_typed() {
        for bytes in [
            vec![0],
            vec![8, 128],
            vec![8, 128, 0],
            vec![8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 2],
            vec![58, 255, 255, 255, 255, 255, 255, 255, 255, 127],
            vec![13, 0, 0, 0, 0],
            vec![58, 10, 1],
        ] {
            assert!(inspect_graph(&bytes, budget()).is_err());
        }
        assert!(matches!(
            Wire::new(&[], MAX_DEPTH + 1),
            Err(PackingGraphBodyError::ResourceExceeded("message_depth"))
        ));
        let raw = Fixture::new().encode();
        assert!(matches!(
            inspect_graph(
                &raw,
                PackingGraphInspectionBudget {
                    max_inspection_host_bytes: 1,
                    ..budget()
                }
            ),
            Err(PackingGraphBodyError::ResourceExceeded(
                "inspection_host_bytes"
            ))
        ));
        assert!(matches!(
            inspect_graph(
                &raw,
                PackingGraphInspectionBudget {
                    max_wire_fields: 1,
                    ..budget()
                }
            ),
            Err(PackingGraphBodyError::ResourceExceeded("wire_fields"))
        ));
        assert!(matches!(
            inspect_graph(
                &raw,
                PackingGraphInspectionBudget {
                    max_wire_fields: 0,
                    ..budget()
                }
            ),
            Err(PackingGraphBodyError::InvalidBudget)
        ));
        assert!(matches!(
            inspect_graph(
                &raw,
                PackingGraphInspectionBudget {
                    max_inspection_host_bytes: u64::MAX,
                    ..budget()
                }
            ),
            Err(PackingGraphBodyError::InvalidBudget)
        ));
        assert!(matches!(
            inspect_graph(&[], budget()),
            Err(PackingGraphBodyError::ArtifactBounds)
        ));
        assert!(matches!(
            inspect_graph(&vec![0; DEVICE_PACKING_MAX_GRAPH_BYTES + 1], budget()),
            Err(PackingGraphBodyError::ArtifactBounds)
        ));
    }
    #[test]
    fn source_recipe_counts_and_integer_payload_have_independent_bounds() {
        let fixture = Fixture::new();
        assert_eq!(fixture.nodes.len(), 275);
        assert_eq!(fixture.inputs.len(), 387);
        assert_eq!(fixture.outputs.len(), 3);
        assert_eq!(fixture.values.len(), 272);
        assert_eq!(fixture.initializers.len(), 5);
        assert_eq!(5 * size_of::<i64>(), 40);
        let report = inspect_graph(&fixture.encode(), budget()).unwrap();
        assert!(report.wire_fields_and_packed_elements > 275);
        assert!(report.wire_fields_and_packed_elements <= PACKING_GRAPH_MAX_WIRE_FIELDS);
    }
}
