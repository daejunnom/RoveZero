"""Preparation contracts only: no model, optimizer, forward, or training run."""
import copy
import hashlib
import importlib.abc
import sys
import unittest


class NeuralImportsForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "onnx", "onnxruntime"}:
            raise AssertionError("neural execution forbidden in Rovent series contract tests")


_guard = NeuralImportsForbidden()
sys.meta_path.insert(0, _guard)
try:
    from rz_pals_model import rovent_generation as series
finally:
    sys.meta_path.remove(_guard)


def seal(value):
    return hashlib.sha256(value.encode()).hexdigest()


def stage(stage_id, kind, parent, teachers):
    return {"stage_id": stage_id, "kind": kind, "parent_stage_id": parent,
            "teacher_actor_ids": teachers, "round_id": stage_id + "-round",
            "role_updates": {"proposer": 2, "critic": 2, "validator": 0}}


def recipe_value():
    actors = [{"actor_id": family, "family": family, "source_revision": ("1" if family == "stockfish" else "2") * 40,
               **{part + "_sha256": seal(family + part) for part in ("source", "binary", "options", "assets", "rights")}}
              for family in ("stockfish", "reckless")]
    return {"schema": series.RECIPE_DOMAIN, "series_id": "rovent-first", "student_id": "common-student",
            "learner_id": "sole-learner", "source_sha256": seal("source"), "model_semantics_sha256": seal("semantics"),
            "actors": actors, "parameter_ownership": {"pc_shared": "pc", "p_private": "pc", "c_private": "pc",
                "v_private": "v", "nnuec": "nnuec", "membership_sha256": seal("membership")},
            "stages": [stage("common", "common_student", None, ["stockfish", "reckless"]),
                       stage("sf-ft", "teacher_finetune", "common", ["stockfish"]),
                       stage("reckless-ft", "teacher_finetune", "common", ["reckless"]),
                       stage("mixed", "mixed", "sf-ft", ["stockfish", "reckless"]),
                       stage("self", "self", "mixed", [])],
            "opening_splits": {"train-a": "train", "train-b": "train", "held": "holdout", "valid": "validation"},
            "global_holdout_openings": ["held"]}


def origin(name, *, stage_id="common", teacher="stockfish", opening="train-a"):
    return {"input_sha256": seal(name + "input"), "record_sha256": seal(name + "record"),
            "producer_sha256": seal(name + "producer"), "stage_id": stage_id, "round_id": stage_id + "-round",
            "teacher_actor_id": teacher, "opening_id": opening, "game_id": name + "-game",
            "line_genealogy_id": name + "-line", "rules_state_sha256": seal(name + "state"),
            "transposition_sha256": seal(name + "transposition")}


def checkpoint(recipe, records=(), *, stage_id="common", parent=None, complete=True):
    selected = recipe.stage(stage_id)
    return {"schema": series.CHECKPOINT_DOMAIN, "recipe_sha256": recipe.sha256, "stage_id": stage_id,
            "checkpoint_sha256": seal(stage_id + "checkpoint"), "model_sha256": seal(stage_id + "model"),
            "parent_checkpoint_sha256": parent, "parameter_membership_sha256": seal("membership"),
            "split_sha256": recipe.split_sha256, "state_sha256": {part: seal(part) for part in series.STATE_PARTS},
            "completed_updates": dict(zip(series.ROLES, selected.role_updates)) if complete else dict.fromkeys(series.ROLES, 0),
            "pending_microbatches": 0, "lineage_teacher_actor_ids": list(recipe.inherited_teachers(stage_id)),
            "training_input_sha256": sorted({record["input_sha256"] for record in records}),
            "training_origin_sha256": sorted({series.record_origin_digest(recipe, record) for record in records})}


class RoventSeriesTests(unittest.TestCase):
    def setUp(self):
        self.guard = NeuralImportsForbidden()
        sys.meta_path.insert(0, self.guard)

    def tearDown(self):
        sys.meta_path.remove(self.guard)

    def test_common_student_and_later_teacher_ft_keep_both_teacher_exposures(self):
        recipe = series.SeriesRecipe(recipe_value())
        self.assertEqual(recipe.inherited_teachers("sf-ft"), ("reckless", "stockfish"))
        self.assertEqual([s.stage_id for s in recipe.lineage("self")], ["common", "sf-ft", "mixed", "self"])
        self.assertEqual(recipe.stage("self").teacher_actor_ids, ())
        value = recipe.to_dict()
        value["student_id"] = "changed-copy"
        self.assertEqual(recipe.to_dict()["student_id"], "common-student")

    def test_frozen_actor_common_stage_and_optimizer_ownership_reject_aliases(self):
        mutations = [lambda value: value["actors"][1].update(family="stockfish"),
                     lambda value: value["actors"][1].update(source_revision="main"),
                     lambda value: value["parameter_ownership"].update(p_private="second-p-moments"),
                     lambda value: value["parameter_ownership"].update(nnuec="v"),
                     lambda value: value["stages"][0].update(teacher_actor_ids=["stockfish"]),
                     lambda value: value["stages"][1].update(parent_stage_id="reckless-ft"),
                     lambda value: value["stages"][4].update(teacher_actor_ids=["stockfish"]),
                     lambda value: value["stages"][0]["role_updates"].update(proposer=True),
                     lambda value: value.update(schema="rz-pals-train-generation/1"),
                     lambda value: value.update(teacher_learners=["one", "two"])]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = recipe_value()
                mutate(value)
                with self.assertRaises(ValueError):
                    series.SeriesRecipe(value)

    def test_shared_common_fork_copies_weights_and_resets_state_while_resume_keeps_state(self):
        recipe = series.SeriesRecipe(recipe_value())
        original = checkpoint(recipe)
        before = copy.deepcopy(original)
        plans = series.common_teacher_forks(recipe, original)
        self.assertEqual({p["parent_checkpoint_sha256"] for p in plans["forks"]}, {original["checkpoint_sha256"]})
        self.assertEqual({p["model_sha256"] for p in plans["forks"]}, {original["model_sha256"]})
        for plan in plans["forks"]:
            self.assertEqual(plan["reset_state"], list(series.STATE_PARTS))
            self.assertEqual(plan["completed_updates"], dict.fromkeys(series.ROLES, 0))
            self.assertIs(plan["execution_authorized"], False)
        resumed = series.resume_plan(recipe, original, stage_id="common")
        self.assertEqual(resumed["checkpoint"], before)
        self.assertEqual(resumed["reset_state"], [])
        resumed["checkpoint"]["state_sha256"]["rng"] = seal("changed")
        self.assertEqual(original, before)
        with self.assertRaisesRegex(ValueError, "resume cannot change stage"):
            series.resume_plan(recipe, original, stage_id="sf-ft")
        incomplete = checkpoint(recipe, complete=False)
        with self.assertRaisesRegex(ValueError, "completed common/parent checkpoint"):
            series.common_teacher_forks(recipe, incomplete)

    def test_resume_rejects_partial_diagnostic_missing_rng_and_erased_teacher_history(self):
        recipe = series.SeriesRecipe(recipe_value())
        for mutation in (lambda value: value.update(pending_microbatches=1),
                         lambda value: value.update(schema="rz-pals-diagnostic-checkpoint/1"),
                         lambda value: value["state_sha256"].pop("rng"),
                         lambda value: value.update(lineage_teacher_actor_ids=["stockfish"])):
            value = checkpoint(recipe)
            mutation(value)
            with self.assertRaises(ValueError):
                series.resume_plan(recipe, value, stage_id="common")

    def test_original_selected_input_binding_cannot_rewrite_legacy_or_future_targets(self):
        from test_generation_contract import row
        record = row()
        record["input"]["opening_id"] = "train-a"
        record["input"]["game_id"] = "game-a"
        record["input"]["line_genealogy_id"] = "line-a"
        record["input_sha256"] = series.seal_input(record["input"])
        before = copy.deepcopy(record)
        recipe = series.SeriesRecipe(recipe_value())
        value = series.index_original_input(recipe, input_snapshot=record["input"], input_sha256=record["input_sha256"],
            record_sha256=seal("record"), producer_sha256=seal("original-producer"), stage_id="common",
            round_id="common-round", teacher_actor_id="reckless")
        self.assertEqual(record, before)
        self.assertEqual(value["input_sha256"], before["input_sha256"])
        self.assertEqual(value["teacher_actor_id"], "reckless")
        bad = copy.deepcopy(record["input"])
        bad["capture_sequence"] += 1
        with self.assertRaisesRegex(ValueError, "input seal mismatch"):
            series.index_original_input(recipe, input_snapshot=bad, input_sha256=record["input_sha256"],
                record_sha256=seal("record"), producer_sha256=seal("producer"), stage_id="common",
                round_id="common-round", teacher_actor_id="reckless")

    def test_transitive_global_leakage_excludes_ancestors_and_both_teacher_branches(self):
        recipe = series.SeriesRecipe(recipe_value())
        a = origin("a")
        b = origin("b", stage_id="sf-ft")
        c = origin("c", stage_id="reckless-ft", teacher="reckless", opening="held")
        b["line_genealogy_id"] = a["line_genealogy_id"]
        c["rules_state_sha256"] = b["rules_state_sha256"]
        index = series.GlobalLineageIndex(recipe, [a, b, c])
        self.assertEqual(index.rows("train", optimization=True), [])
        self.assertEqual(index.rows("holdout"), [])
        with self.assertRaisesRegex(ValueError, "global holdout/cross-split contamination"):
            index.admit_checkpoint_lineage(recipe, checkpoint(recipe, [a]), ancestors=[])
        with self.assertRaisesRegex(ValueError, "cannot enter optimization"):
            index.rows("holdout", optimization=True)

    def test_distinct_teacher_origins_survive_identical_input_seals_and_global_audit(self):
        recipe = series.SeriesRecipe(recipe_value())
        sf, reckless = origin("sf"), origin("reckless", teacher="reckless")
        reckless["input_sha256"] = sf["input_sha256"]
        rows = [sf, reckless]
        index = series.GlobalLineageIndex(recipe, rows)
        self.assertEqual(len(index.rows("train", optimization=True)), 2)
        self.assertNotEqual(series.record_origin_digest(recipe, sf), series.record_origin_digest(recipe, reckless))
        result = index.admit_checkpoint_lineage(recipe, checkpoint(recipe, rows), ancestors=[])
        self.assertEqual((result["training_origins"], result["training_inputs"]), (2, 1))
        self.assertIs(result["native_intake_verified"], False)
        self.assertIs(result["execution_authorized"], False)

    def test_checkpoint_chain_cannot_erase_ancestor_or_consume_sibling_teacher_rows(self):
        recipe = series.SeriesRecipe(recipe_value())
        common, sf, reckless = origin("common"), origin("sf", stage_id="sf-ft"), origin("reckless", stage_id="reckless-ft", teacher="reckless")
        index = series.GlobalLineageIndex(recipe, [common, sf, reckless])
        parent = checkpoint(recipe, [common])
        child = checkpoint(recipe, [common, sf], stage_id="sf-ft", parent=parent["checkpoint_sha256"])
        result = index.admit_checkpoint_lineage(recipe, child, ancestors=[parent])
        self.assertEqual(result["training_origins"], 2)
        with self.assertRaisesRegex(ValueError, "all common/branch ancestor"):
            index.admit_checkpoint_lineage(recipe, child, ancestors=[])
        erased = checkpoint(recipe, [sf], stage_id="sf-ft", parent=parent["checkpoint_sha256"])
        with self.assertRaisesRegex(ValueError, "erased ancestor"):
            index.admit_checkpoint_lineage(recipe, erased, ancestors=[parent])
        sibling = checkpoint(recipe, [common, reckless], stage_id="sf-ft", parent=parent["checkpoint_sha256"])
        with self.assertRaisesRegex(ValueError, "future or sibling branch"):
            index.admit_checkpoint_lineage(recipe, sibling, ancestors=[parent])


if __name__ == "__main__":
    unittest.main()
