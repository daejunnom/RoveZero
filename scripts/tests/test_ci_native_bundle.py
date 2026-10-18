"""Pure metadata/FD custody cases; no Cargo, target ELF or neural execution."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ci_native_bundle as bundle


class NativeBundleTests(unittest.TestCase):
    def test_compile_only_cleanup_preserves_published_bundle_and_unique_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / 'artifact'
            artifact.mkdir()
            archive = artifact / 'native-validation-binaries.tar.gz'
            archive.write_bytes(b'synthetic already-verified archive')
            (artifact / 'only-game.pgn').write_bytes(b'unique-result')
            (root / 'unclassified-other-data').mkdir()
            for name in ('target-native', 'private-elf', 'cargo-home', 'logs', 'tmp'):
                (root / name).mkdir()
                (root / name / 'owned-copy').write_bytes(b'regenerable')
            receipt = {'status': 'SAME_HEAD_RELEASE_ELFS_AND_STATIC_SOURCE_METADATA_VERIFIED',
                       'commands': [{'original_tree_cleanup_verified': True}],
                       'binary_bundle': bundle.file_pin(archive)}
            journal = bundle.cleanup_compiler_outputs(root, artifact, receipt)
            self.assertTrue(journal['complete'])
            self.assertTrue((root / 'logs' / 'owned-copy').exists())
            self.assertFalse((root / 'private-elf').exists())
            self.assertFalse((root / 'target-native').exists())
            self.assertEqual(archive.read_bytes(), b'synthetic already-verified archive')
            self.assertEqual((artifact / 'only-game.pgn').read_bytes(), b'unique-result')
            self.assertTrue((root / 'unclassified-other-data').exists())

    def test_uncertain_tree_or_changed_bundle_prevents_all_compiler_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / 'artifact'
            artifact.mkdir()
            archive = artifact / 'native-validation-binaries.tar.gz'
            archive.write_bytes(b'retained')
            (root / 'target-native').mkdir()
            receipt = {'status': 'SAME_HEAD_RELEASE_ELFS_AND_STATIC_SOURCE_METADATA_VERIFIED',
                       'commands': [{'original_tree_cleanup_verified': False}],
                       'binary_bundle': bundle.file_pin(archive)}
            with self.assertRaisesRegex(RuntimeError, 'VerifiedBundleAndTreeClosure'):
                bundle.cleanup_compiler_outputs(root, artifact, receipt)
            receipt['commands'][0]['original_tree_cleanup_verified'] = True
            archive.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'VerifiedBundleAndTreeClosure'):
                bundle.cleanup_compiler_outputs(root, artifact, receipt)
            self.assertTrue((root / 'target-native').exists())
            self.assertNotIn('regenerable_cleanup', receipt)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux-only bundle FD/ctime custody; no Windows emulation')
    def test_cargo_hardlink_is_copied_into_single_link_custody(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, alias, private = root / 'cargo', root / 'deps-alias', root / 'private'
            source.write_bytes(b'\x7fELF_HEADER_ONLY_SYNTHETIC_NOT_EXECUTABLE')
            os.link(source, alias)
            with self.assertRaisesRegex(RuntimeError, 'ArtifactTypeOrSize'):
                bundle.file_pin(source)
            budgets = []
            copied = bundle.copy_cargo_elf(source, private, budgets.append)
            self.assertEqual(copied['original_cargo']['nlink'], 2)
            self.assertEqual(private.stat().st_nlink, 1)
            self.assertEqual(bundle.file_pin(private), {k: copied['original_cargo'][k] for k in ('bytes', 'sha256')})
            self.assertEqual(budgets[0], source.stat().st_size)
            with self.assertRaisesRegex(RuntimeError, 'CargoElfCopyPlacement'):
                bundle.copy_cargo_elf(source, private, budgets.append)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux-only bundle FD/ctime custody; no Windows emulation')
    def test_non_elf_and_copy_admission_failure_are_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, private = root / 'source', root / 'private'
            source.write_bytes(b'NOT_ELF')
            with self.assertRaisesRegex(RuntimeError, 'CargoOutputNotElf'):
                bundle.copy_cargo_elf(source, private, lambda _: None)
            self.assertFalse(private.exists())
            source.write_bytes(b'\x7fELFfixture')
            def reject(_):
                raise RuntimeError('BudgetDenied')
            with self.assertRaisesRegex(RuntimeError, 'BudgetDenied'):
                bundle.copy_cargo_elf(source, private, reject)
            self.assertFalse(private.exists())

    def test_explicit_source_paths_reject_secrets_heavy_and_traversal(self):
        for path in ('crates/x/.env', 'scripts/credentials.json', 'crates/x/models/w.bin',
                     'scripts/../x.py', '/scripts/x.py', 'scripts\\x.py'):
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, 'ForbiddenSourcePath'):
                bundle.allowed_source(path)
        self.assertFalse(bundle.allowed_source('docs/research/PALS-IMPLEMENTATION.md'))
        self.assertTrue(bundle.allowed_source('scripts/ci_native_bundle.py'))

    @unittest.skipUnless(sys.platform == 'linux', 'Linux-only bundle FD/ctime custody; no Windows emulation')
    def test_source_bytes_must_match_exact_tracked_git_blob(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            tree = []
            for logical in sorted(bundle.SINGLETONS | {bundle.CONTROL}):
                path = source / logical
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = b'' if logical == 'LICENSE' else b'SYNTHETIC_PUBLIC_SOURCE\n'
                path.write_bytes(raw)
                oid = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
                tree.append(b'100644 blob ' + oid.encode() + b'\t' + logical.encode() + b'\0')
            roster = b''.join(tree)
            rows, controls = bundle.source_inventory(source, roster)
            self.assertEqual(len(rows), len(bundle.SINGLETONS))
            self.assertEqual(set(controls), {bundle.CONTROL})
            (source / 'Cargo.lock').write_bytes(b'CHANGED')
            with self.assertRaisesRegex(RuntimeError, 'CheckoutBytesDifferFromGitBlob'):
                bundle.source_inventory(source, roster)

    def test_component_framing_is_order_independent_and_content_sensitive(self):
        names = list(bundle.CORE5) + ['component' + str(n) for n in range(20)]
        rows = [{'path': 'crates/b.rs', 'bytes': 2, 'sha256': 'ab' * 32},
                {'path': 'crates/a.rs', 'bytes': 1, 'sha256': 'cd' * 32}]
        catalog = {'domain': 'rz-pals-source-inventory-v5/1',
                   'algorithm': 'sha256-domain-u64le;sorted-path-u64le-utf8;size-u64le;sha256-raw32',
                   'components': {name: [row['path'] for row in rows] for name in names}}
        initial = bundle.component_inventory(catalog, rows)
        reversed_catalog = copy.deepcopy(catalog)
        for paths in reversed_catalog['components'].values():
            paths.reverse()
        self.assertEqual(initial, bundle.component_inventory(reversed_catalog, list(reversed(rows))))
        changed = copy.deepcopy(rows)
        changed[0]['bytes'] += 1
        self.assertNotEqual(initial, bundle.component_inventory(catalog, changed))
        catalog['components'][names[0]].append(rows[0]['path'])
        with self.assertRaisesRegex(RuntimeError, 'CatalogSourceMissing'):
            bundle.component_inventory(catalog, rows)

    def test_warm_requires_both_registered_source_and_explicit_exact_feature_target(self):
        name = bundle.WARM[2]
        features = bundle.WARM[3].split(',')
        logical = 'crates/rz-uci/examples/' + name + '.rs'
        manifest = {'example': [{'name': name, 'required-features': features}],
                    'features': {feature: [] for feature in features}}
        spec, status = bundle.warm_spec(manifest, {logical})
        self.assertEqual(spec, bundle.WARM)
        self.assertEqual(status, 'SOURCE_PRESENT_REQUIRES_ACTUAL_BUILD')
        self.assertIsNone(bundle.warm_spec(manifest, set())[0])
        self.assertIsNone(bundle.warm_spec({'features': manifest['features']}, {logical})[0])
        manifest['example'][0]['required-features'] = ['onnx-cpu']
        self.assertEqual(bundle.warm_spec(manifest, {logical})[1], 'NOT_READY_REQUIRED_FEATURE_OR_SOURCE_CONTRACT')

    def test_actual_cargo_artifact_must_join_source_kind_features_and_separate_slot(self):
        source = Path('/synthetic-source')
        target_root = Path('/synthetic-target-diagnostic')
        spec = bundle.SPECS[-1]
        logical = 'crates/rz-uci/examples/pals_continuation_diagnostic.rs'
        target = {'name': spec[2], 'kind': ['example'], 'crate_types': ['bin'],
                  'src_path': str(source / logical), 'edition': '2024'}
        metadata = {'packages': [{'name': 'rz-uci', 'id': 'SYNTHETIC_PACKAGE',
                                 'manifest_path': str(source / 'crates/rz-uci/Cargo.toml'), 'targets': [target]}]}
        artifact = {'reason': 'compiler-artifact', 'package_id': 'SYNTHETIC_PACKAGE',
                    'target': copy.deepcopy(target), 'executable': str(target_root / 'release/examples' / spec[2]),
                    'profile': {'test': False},
                    'features': ['onnx-cpu', 'pals-adaptive', 'pals-adaptive-diagnostics', 'search-work-receipts']}
        messages = [artifact, {'reason': 'build-finished', 'success': True}]
        def check(value, paths=None):
            raw = b'\n'.join(json.dumps(row).encode() for row in value)
            return bundle.cargo_artifact(raw, metadata, source, target_root, spec,
                                         {logical} if paths is None else paths)
        self.assertEqual(check(messages)[0], Path(artifact['executable']))
        for label, mutate in (
                ('duplicate', lambda v: v.insert(0, copy.deepcopy(v[0]))),
                ('failed-finish', lambda v: v[1].update(success=False)),
                ('wrong-slot', lambda v: v[0].update(executable='/wrong-slot/diagnostic')),
                ('wrong-package', lambda v: v[0].update(package_id='OTHER')),
                ('test-profile', lambda v: v[0]['profile'].update(test=True)),
                ('default', lambda v: v[0]['features'].append('default')),
                ('warm-mixed', lambda v: v[0]['features'].append('experimental-io-binding'))):
            changed = copy.deepcopy(messages)
            mutate(changed)
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                check(changed)
        with self.assertRaisesRegex(RuntimeError, 'ExactRegisteredCargoTarget'):
            check(messages, set())

    def test_strict_metadata_rejects_duplicate_keys_and_nonfinite_values(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(RuntimeError):
                bundle.strict_json(raw)

    def test_rovent_compile_metadata_cannot_claim_native_execution_or_replace_registered_sources(self):
        components = {'uci': 'ab' * 32, 'rules': 'cd' * 32}
        head = '19' * 20
        value = {'domain': 'rz-rovent-build-capabilities/1', 'source_commit': head,
                 'source_components': dict(components, rovent_bootstrap='ef' * 32,
                                           rovent_process_boundary='42' * 32),
                 'target_os': 'linux', 'target_arch': 'x86_64',
                 'onnx_cpu_compiled': True, 'onnx_cuda_compiled': True,
                 'native_runtime_loaded': False, 'native_model_loaded': False,
                 'neural_execution_observed': False}
        bundle.validate_rovent_build_metadata(value, head, components)
        for field in ('native_runtime_loaded', 'native_model_loaded', 'neural_execution_observed'):
            wrong = copy.deepcopy(value)
            wrong[field] = True
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'StaticRoventBuildMetadata'):
                bundle.validate_rovent_build_metadata(wrong, head, components)
        for name in ('uci', 'rules', 'rovent_bootstrap', 'rovent_process_boundary'):
            wrong = copy.deepcopy(value)
            wrong['source_components'][name] = 'NOT_A_REGISTERED_DIGEST'
            with self.subTest(component=name), self.assertRaisesRegex(RuntimeError, 'StaticRoventRegisteredSourceMismatch'):
                bundle.validate_rovent_build_metadata(wrong, head, components)

    def test_registry_public_methods_remain_static_and_source_bound_elsewhere(self):
        registry = {'schema': 'rz-pals-report-model-registry-query/1', 'model_profile': 'virtual_review_v1',
                    'registry_methods': ['PalsModelProfile::VirtualReviewV1.registered_semantics_sha256',
                                         'pals_model::virtual_review::registered_encoding_sha256'],
                    'model_semantic_sha256': 'ab' * 32, 'encoding_sha256': 'cd' * 32,
                    'numerical_authority_issued': False, 'model_initialized': False,
                    'ort_loaded': False, 'neural_executed': False, 'training_executed': False}
        bundle.validate_registry(registry)
        for key in ('numerical_authority_issued', 'model_initialized', 'ort_loaded', 'neural_executed', 'training_executed'):
            changed = copy.deepcopy(registry)
            changed[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'StaticRegistryGetterOutput'):
                bundle.validate_registry(changed)
        registry['model_semantic_sha256'] = 'not-an-actual-registry-digest'
        with self.assertRaisesRegex(RuntimeError, 'StaticRegistryGetterOutput'):
            bundle.validate_registry(registry)


if __name__ == '__main__':
    unittest.main()
