//! Real joined-owner -> same preparation actor release regressions.
//! Registered as a child of deferred_value_tests to reuse its small explicit
//! model fixture, not a second model/TT implementation.

use super::*;
use crate::cpu_value::nnuec::v2::retirement::{NnuecCacheInvalidation, NnuecCacheReleaseAck};
use crate::cpu_worker::{CpuWorker, NnuecEngineReleaseAck, RecoveredNnuecEngine};

fn ready_worker(
    fixture: &Fixture,
    source: &DeferredNnuecRSource,
) -> (CpuWorker, NnuecReadyRegistration) {
    let starting = start(source, materialize(request(source, fixture)));
    wait_ready(&starting);
    starting
        .accept_ready()
        .unwrap_or_else(|_| panic!("actual ready admission"))
}

fn join_worker(worker: &mut CpuWorker) -> crate::cpu_worker::CpuWorkerShutdown {
    worker.request_shutdown();
    let until = Instant::now() + WINDOW;
    loop {
        let observed = worker.signal.version();
        if let Some(receipt) = worker.try_join().unwrap() {
            return receipt;
        }
        assert!(Instant::now() < until);
        worker
            .signal
            .wait_changed(observed, Duration::from_millis(10));
    }
}

fn release(actor: &PreparationActor, owner: RecoveredNnuecEngine) -> NnuecEngineReleaseAck {
    let id = actor
        .try_retire_engine(owner)
        .unwrap_or_else(|_| panic!("retirement admission"));
    let until = Instant::now() + WINDOW;
    loop {
        if let Some(completion) = actor.try_retired_engine().unwrap() {
            assert_eq!(completion.request_id(), &id);
            return completion
                .into_result()
                .unwrap_or_else(|(_, error)| panic!("release failed: {error:?}"));
        }
        assert!(Instant::now() < until);
        std::thread::yield_now();
    }
}

fn cache_release(
    actor: &PreparationActor,
    ticket: &NnuecCacheInvalidation,
) -> NnuecCacheReleaseAck {
    let until = Instant::now() + WINDOW;
    loop {
        if let Some(ack) = actor.try_context_cache_release(ticket).unwrap() {
            return ack;
        }
        assert!(Instant::now() < until);
        std::thread::yield_now();
    }
}

#[test]
fn actual_joined_engine_releases_original_evaluator_but_not_other_view_custody() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let (mut worker, registration) = ready_worker(&fixture, &source);
    let profile = registration.profile();
    let stack = registration.actual_stack_owner_id();
    let binding = registration.binding_digest();
    drop(registration);
    let stopped = join_worker(&mut worker);
    assert!(stopped.thread_joined && stopped.engine_recoverable);
    let owner = worker.take_nnuec_retirement().unwrap();
    assert!(owner.matches_source(&source));
    assert_eq!(owner.actual_stack_owner_id(), stack);
    let before = Arc::strong_count(&fixture.bound);
    let ack = release(&fixture.actor, owner);
    assert!(ack.matches_source(&source));
    assert_eq!(ack.profile(), profile);
    assert_eq!(ack.binding_digest(), binding);
    assert_eq!(ack.actual_stack_owner_id(), stack);
    assert_eq!(ack.model_identity(), fixture.model.identity());
    assert_eq!(ack.provenance(), fixture.bound.provenance());
    assert_eq!(Arc::strong_count(&fixture.bound), before - 2);
    assert!(
        fixture.bound.view().validates(&fixture.stores),
        "other original custody still pins the view"
    );
}

#[test]
fn active_worker_cannot_transfer_tt_and_actual_join_allows_only_one_transfer() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let (mut worker, registration) = ready_worker(&fixture, &source);
    assert!(matches!(
        worker.take_nnuec_retirement(),
        Err(CpuWorkerError::AlreadyReserved)
    ));
    let stopped = join_worker(&mut worker);
    assert!(stopped.engine_recoverable);
    let owner = worker.take_nnuec_retirement().unwrap();
    assert!(matches!(
        worker.take_nnuec_retirement(),
        Err(CpuWorkerError::Duplicate)
    ));
    assert_eq!(owner.binding_digest(), registration.binding_digest());
    release(&fixture.actor, owner);
}

#[test]
fn equal_model_and_configuration_do_not_substitute_another_source_authority() {
    let fixture = fixture();
    let original = source(&fixture, false);
    let equivalent = source(&fixture, false);
    let (mut worker, registration) = ready_worker(&fixture, &original);
    join_worker(&mut worker);
    let owner = worker.take_nnuec_retirement().unwrap();
    assert!(owner.matches_source(&original));
    assert!(!owner.matches_source(&equivalent));
    let ack = release(&fixture.actor, owner);
    assert!(ack.matches_source(&original));
    assert!(ack.matches_source(&original.clone()));
    assert!(!ack.matches_source(&equivalent));
    drop(registration);
}

#[test]
fn a_foreign_actor_returns_same_actual_engine_for_the_original_actor() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let (mut worker, registration) = ready_worker(&fixture, &source);
    join_worker(&mut worker);
    let owner = worker.take_nnuec_retirement().unwrap();
    let stack = owner.actual_stack_owner_id();
    let foreign = PreparationActor::spawn(Arc::clone(&fixture.model)).unwrap();
    let id = foreign
        .try_retire_engine(owner)
        .unwrap_or_else(|_| panic!("foreign request admission"));
    let until = Instant::now() + WINDOW;
    let owner = loop {
        if let Some(completion) = foreign.try_retired_engine().unwrap() {
            assert_eq!(completion.request_id(), &id);
            let (owner, error) = completion.into_result().unwrap_err();
            assert!(matches!(error, CpuWorkerError::Invalid(_)));
            break owner;
        }
        assert!(Instant::now() < until);
        std::thread::yield_now();
    };
    assert_eq!(owner.actual_stack_owner_id(), stack);
    assert_eq!(owner.binding_digest(), registration.binding_digest());
    assert_eq!(
        release(&fixture.actor, owner).actual_stack_owner_id(),
        stack
    );
    foreign.shutdown().unwrap();
}

#[test]
fn canceled_and_cache_invalidated_bound_still_has_cleanup_only_owner_rights() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let (mut worker, registration) = ready_worker(&fixture, &source);
    fixture.cancel.store(true, Ordering::Release);
    let epoch = fixture.actor.invalidate_cache().unwrap();
    let until = Instant::now() + WINDOW;
    while fixture.actor.cache_invalidated_through() < epoch {
        assert!(Instant::now() < until);
        std::thread::yield_now();
    }
    assert!(!fixture.bound.authority_is_live());
    assert!(join_worker(&mut worker).engine_recoverable);
    let owner = worker.take_nnuec_retirement().unwrap();
    let ack = release(&fixture.actor, owner);
    assert_eq!(ack.binding_digest(), registration.binding_digest());
    assert!(
        !fixture.bound.authority_is_live(),
        "cleanup never restores score or resume authority"
    );
}

#[test]
fn busy_preparation_returns_same_engine_without_consuming_or_dropping_it() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let (mut worker, registration) = ready_worker(&fixture, &source);
    join_worker(&mut worker);
    let owner = worker.take_nnuec_retirement().unwrap();
    let request = Request::admit(
        &fixture.model,
        Arc::clone(&fixture.view),
        &fixture.stores,
        fixture.bound.provenance().root,
        fixture.bound.provenance().generation,
        Instant::now() + WINDOW,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    fixture
        .actor
        .try_submit(request)
        .unwrap_or_else(|_| panic!("prepare admission"));
    let (error, owner) = fixture.actor.try_retire_engine(owner).unwrap_err();
    assert_eq!(
        error,
        crate::cpu_value::nnuec::v2::review::PreparationError::Busy
    );
    assert_eq!(owner.binding_digest(), registration.binding_digest());
    let until = Instant::now() + WINDOW;
    loop {
        if let Some(completion) = fixture.actor.try_complete().unwrap() {
            drop(completion.result.unwrap());
            break;
        }
        assert!(Instant::now() < until);
        std::thread::yield_now();
    }
    release(&fixture.actor, owner);
}

#[test]
fn typed_cache_ack_is_actor_owned_single_use_and_separate_from_engine_release() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let (mut worker, registration) = ready_worker(&fixture, &source);
    let ticket = fixture.actor.invalidate_context_cache().unwrap();
    let foreign = PreparationActor::spawn(Arc::clone(&fixture.model)).unwrap();
    assert!(foreign.try_context_cache_release(&ticket).is_err());
    let cache = cache_release(&fixture.actor, &ticket);
    assert_eq!(cache.epoch(), ticket.epoch());
    assert!(fixture.actor.try_context_cache_release(&ticket).is_err());
    join_worker(&mut worker);
    let owner = worker.take_nnuec_retirement().unwrap();
    let ack = release(&fixture.actor, owner);
    assert!(ack.matches_cache_release(&cache));
    assert!(!fixture.bound.authority_is_live());
    assert_eq!(ack.binding_digest(), registration.binding_digest());
    foreign.shutdown().unwrap();
}

#[test]
fn a_different_actor_cache_ack_cannot_release_this_actors_root_context() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let (mut worker, registration) = ready_worker(&fixture, &source);
    let foreign = PreparationActor::spawn(Arc::clone(&fixture.model)).unwrap();
    let ticket = foreign.invalidate_context_cache().unwrap();
    let foreign_cache = cache_release(&foreign, &ticket);
    join_worker(&mut worker);
    let owner = worker.take_nnuec_retirement().unwrap();
    let ack = release(&fixture.actor, owner);
    assert!(!ack.matches_cache_release(&foreign_cache));
    assert_eq!(ack.binding_digest(), registration.binding_digest());
    foreign.shutdown().unwrap();
}

#[test]
fn a_previous_clear_ack_does_not_release_a_later_prepared_bound() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let old_ticket = fixture.actor.invalidate_context_cache().unwrap();
    let old_cache = cache_release(&fixture.actor, &old_ticket);
    let request = Request::admit(
        &fixture.model,
        Arc::clone(&fixture.view),
        &fixture.stores,
        fixture.bound.provenance().root,
        fixture.bound.provenance().generation,
        Instant::now() + WINDOW,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    fixture
        .actor
        .try_submit(request)
        .unwrap_or_else(|_| panic!("new epoch prepare"));
    let until = Instant::now() + WINDOW;
    let bound = loop {
        if let Some(completion) = fixture.actor.try_complete().unwrap() {
            break Arc::new(completion.result.unwrap());
        }
        assert!(Instant::now() < until);
        std::thread::yield_now();
    };
    let request = source
        .prepare_materialization(
            Arc::clone(&bound),
            &fixture.stores,
            bound.provenance().view_seal,
        )
        .unwrap();
    let starting = start(&source, materialize(request));
    wait_ready(&starting);
    let (mut worker, registration) = starting.accept_ready().unwrap_or_else(|_| panic!("ready"));
    join_worker(&mut worker);
    let owner = worker.take_nnuec_retirement().unwrap();
    let ack = release(&fixture.actor, owner);
    assert!(!ack.matches_cache_release(&old_cache));
    let current_ticket = fixture.actor.invalidate_context_cache().unwrap();
    let current_cache = cache_release(&fixture.actor, &current_ticket);
    assert!(ack.matches_cache_release(&current_cache));
    assert_eq!(ack.binding_digest(), registration.binding_digest());
}

#[test]
fn final_installed_stack_owner_is_released_instead_of_initial_ready_owner() {
    let fixture = fixture();
    let source = source(&fixture, false);
    let (mut worker, initial) = ready_worker(&fixture, &source);
    let payload = crate::cpu_worker::PreparedValueInstallation::from_bound(
        Arc::clone(&fixture.model),
        Arc::clone(&fixture.bound),
        &fixture.stores,
        fixture.bound.provenance().view_seal,
    )
    .unwrap();
    let install_id = worker
        .try_install_prepared_value(payload)
        .unwrap_or_else(|_| panic!("actual idle installation admission"));
    let until = Instant::now() + WINDOW;
    let installed = loop {
        if let Some(receipt) = worker.try_install_readback().unwrap() {
            assert_eq!(receipt.installation_id(), install_id);
            break receipt;
        }
        assert!(Instant::now() < until);
        std::thread::yield_now();
    };
    let actual = installed.result().unwrap();
    assert_ne!(
        actual.actual_stack_owner_id(),
        initial.actual_stack_owner_id()
    );
    join_worker(&mut worker);
    let owner = worker.take_nnuec_retirement().unwrap();
    assert_eq!(
        owner.actual_stack_owner_id(),
        actual.actual_stack_owner_id()
    );
    let ack = release(&fixture.actor, owner);
    assert_eq!(ack.actual_stack_owner_id(), actual.actual_stack_owner_id());
    assert_eq!(ack.profile(), actual.profile());
    assert_eq!(ack.binding_digest(), installed.binding_digest());
    drop(initial);
    let invalidation = fixture.actor.invalidate_context_cache().unwrap();
    let cache = cache_release(&fixture.actor, &invalidation);
    assert!(ack.matches_cache_release(&cache));
    let weak_bound = Arc::downgrade(&fixture.bound);
    drop(fixture.bound);
    drop(fixture.view);
    assert!(weak_bound.upgrade().is_none());
    assert_eq!(fixture.stores.review_pin_count().unwrap(), 0);
    assert!(
        installed.is_installed(),
        "receipt remains without holding view pins"
    );
}
