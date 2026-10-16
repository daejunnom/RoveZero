//! Source-only archive fixture: no native input, model, collection, optimizer
//! or actual-generation eligibility is exercised by these dummy source pins.
use crate::pals_collect_v3::{
    GenerationChunkKindV3, GenerationChunkPinV3, GenerationWriteCommandV3, GenerationWriterEventV3,
    GenerationWriterWorkerV3, verify_chunk,
};
use rz_experiments::pals_data_v3::{
    GenerationProducerAuthorityV3, GenerationSplitV3, generation_digest,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);

#[test]
fn preserve_partial_archive_reads_back_failure_bytes_and_joins_without_complete_manifest() {
    let temp_root = fs::canonicalize(std::env::temp_dir()).unwrap();
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let requested_root = temp_root.join(format!(
        "rz-generation-partial-source-fixture-{}-{suffix}-{}",
        std::process::id(),
        NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
    ));
    // create_dir never adopts an existing directory owned by another task.
    fs::create_dir(&requested_root).unwrap();
    let owned_root = fs::canonicalize(&requested_root).unwrap();
    assert_eq!(owned_root.parent(), Some(temp_root.as_path()));
    assert_eq!(owned_root.file_name(), requested_root.file_name());
    let archive = owned_root.join("archive");
    assert!(!archive.exists());
    let until = Instant::now() + Duration::from_secs(10);
    let mut worker = GenerationWriterWorkerV3::spawn(
        archive.clone(),
        "source-fixture-partial-archive".into(),
        32 * 1024 * 1024,
        until,
    )
    .unwrap();

    loop {
        match worker.try_event() {
            Some(GenerationWriterEventV3::Ready) => break,
            Some(GenerationWriterEventV3::Failed(error)) => {
                panic!("source archive fixture preparation: {error}")
            }
            Some(_) => panic!("source archive fixture finalized before requested preservation"),
            None => {
                assert!(Instant::now() < until, "source fixture readiness deadline");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    let raw_failure = serde_json::json!({
        "schema": "rz-pals-generation-game-failure-evidence/1",
        "game_id": "source-fixture-game",
        "opening_id": "source-fixture-opening",
        "stage": "source_fixture",
        "error": "Archive source fixture only; no actual engine or collection was executed."
    });
    let evidence_sha =
        generation_digest("rz-pals-generation-game-failure-evidence/1", &raw_failure).unwrap();
    assert!(
        worker
            .try_submit(GenerationWriteCommandV3::Evidence(raw_failure.clone()))
            .is_ok()
    );
    let splits = BTreeMap::from([("source-fixture-opening".into(), GenerationSplitV3::Train)]);
    // Valid syntax is required to reach the archive code. These dummy hashes,
    // empty neural rows and fixture IDs never claim a real producer execution.
    let authority = GenerationProducerAuthorityV3 {
        schema: "rz-pals-actual-collection-authority/1".into(),
        source_kind: "actual_selfplay".into(),
        source_sha256: "1".repeat(64),
        model_sha256: "2".repeat(64),
        runtime_sha256: "3".repeat(64),
        teacher_binary_sha256: "4".repeat(64),
        teacher_profile_sha256: "5".repeat(64),
        opening_split_sha256: generation_digest("rz-pals-opening-split/1", &splits).unwrap(),
        registered_before_outcomes: true,
        input_seals: vec![],
        actual_game_ids: vec![],
    };
    let reason = "source fixture: interruption after queued failure evidence";
    assert!(
        worker
            .try_submit(GenerationWriteCommandV3::PreservePartial {
                authority,
                splits,
                reason: reason.into(),
            })
            .is_ok()
    );
    // Terminal admission must close immediately. A late row stays owned by the
    // caller rather than being dropped or reported as part of this archive.
    match worker.try_submit(GenerationWriteCommandV3::Evidence(raw_failure.clone())) {
        Err((_, GenerationWriteCommandV3::Evidence(rejected))) => {
            assert_eq!(rejected, raw_failure)
        }
        _ => panic!("source fixture accepted or lost a post-preservation command"),
    }

    let partial = loop {
        match worker.try_event() {
            Some(GenerationWriterEventV3::PartialPreserved(partial)) => break partial,
            Some(GenerationWriterEventV3::Failed(error)) => {
                panic!("source archive fixture preservation: {error}")
            }
            Some(GenerationWriterEventV3::Finished { .. }) => {
                panic!("partial preservation was mislabeled as a completed generation")
            }
            Some(GenerationWriterEventV3::Ready) => panic!("duplicate source fixture readiness"),
            None => {
                assert!(
                    Instant::now() < until,
                    "source fixture preservation deadline"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    };
    assert_eq!(partial.directory, fs::canonicalize(&archive).unwrap());
    assert_eq!(partial.directory.parent(), Some(owned_root.as_path()));
    assert_eq!(
        partial.manifest_path,
        partial.directory.join("manifest.partial.json")
    );
    assert_eq!(partial.reason, reason);
    let manifest_bytes = fs::read(&partial.manifest_path).unwrap();
    assert!(manifest_bytes.ends_with(b"\n"));
    assert_eq!(
        format!("{:x}", Sha256::digest(&manifest_bytes)),
        partial.manifest_sha256
    );
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(
        manifest["schema"],
        "rz-pals-generation-partial-chunk-manifest/1"
    );
    assert_eq!(manifest["source_kind"], "actual_selfplay_partial");
    assert_eq!(manifest["incomplete_reason"], reason);
    assert_eq!(manifest["total_records"], 0);
    let manifest_pins: Vec<GenerationChunkPinV3> =
        serde_json::from_value(manifest["chunks"].clone()).unwrap();
    assert_eq!(manifest_pins, partial.chunks);
    assert_eq!(partial.chunks.len(), 1);
    let pin = &partial.chunks[0];
    assert_eq!(pin.kind, GenerationChunkKindV3::Evidence);
    assert_eq!(pin.rows, 1);
    verify_chunk(&partial.directory, pin).unwrap();
    let actual_chunk = fs::read(partial.directory.join(&pin.path)).unwrap();
    assert_eq!(actual_chunk.len() as u64, pin.bytes);
    assert_eq!(manifest["total_bytes"], pin.bytes);
    assert_eq!(format!("{:x}", Sha256::digest(&actual_chunk)), pin.sha256);
    let persisted: serde_json::Value = serde_json::from_slice(&actual_chunk).unwrap();
    assert_eq!(
        persisted,
        serde_json::json!({"sha256": evidence_sha, "receipt": raw_failure})
    );
    assert!(!partial.directory.join("manifest.json").exists());
    assert!(!partial.directory.join("manifest.pending.json").exists());
    assert!(
        !partial
            .directory
            .join("manifest.partial.pending.json")
            .exists()
    );

    loop {
        if worker.try_join().unwrap() {
            break;
        }
        assert!(
            Instant::now() < until,
            "source fixture archive join deadline"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(worker.try_join().unwrap());
    drop(worker);
    // Only this fresh fixture's exact canonical child is deleted after actual
    // thread join. Production recovery/deletion and training admission are not
    // exercised by this test; failed assertions leave evidence for inspection.
    assert_eq!(partial.directory, fs::canonicalize(&archive).unwrap());
    assert_eq!(partial.directory.parent(), Some(owned_root.as_path()));
    assert!(partial.directory.join("owner.json").exists());
    fs::remove_dir_all(&partial.directory).unwrap();
    assert_eq!(owned_root, fs::canonicalize(&requested_root).unwrap());
    assert_eq!(owned_root.parent(), Some(temp_root.as_path()));
    fs::remove_dir(owned_root).unwrap();
}
