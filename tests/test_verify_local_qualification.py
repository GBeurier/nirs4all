"""Source/proof tamper and incomplete-host refusals in the shared verifier."""

import copy
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("qualification", Path(__file__).resolve().parents[1] / "scripts/verify_local_qualification.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LocalQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", self.root], check=True)
        (self.root / "bindings").mkdir()
        (self.root / "bindings/runtime.rs").write_text("// genuine test fixture, not a product qualification\n")
        (self.root / "Cargo.toml").write_text('[package]\nname = "unit-fixture"\nversion = "1.0.0"\n')
        (self.root / "qualification").mkdir()
        (self.root / "qualification/native.bin").write_bytes(b"Synthetic unit-test native artifact, not product bytes.")
        artifact = {"name": "unit-fixture", "version": "1", "origin": "unit-test-only", "sha256": MODULE.digest(self.root / "qualification/native.bin"), "bytes": (self.root / "qualification/native.bin").stat().st_size, "artifact_path": "qualification/native.bin"}
        policy = {
            "schema": MODULE.SCHEMA,
            "project": "core",
            "gates": [
                {
                    "id": "parity",
                    "hosts": ["linux", "windows"],
                    "minimum_passed": 3,
                    "facts": [{"field": "hpo_trials", "op": "eq", "value": 45}, {"field": "max_abs_error", "op": "max", "value": 1e-8}],
                    "requires_provenance": True,
                    "provenance_requirements": {k: [artifact] for k in ("source_artifacts", "dependency_origins")},
                    "input_paths": ["Cargo.toml", "bindings/"],
                    "commands": {host: ["unit-fixture-probe"] for host in ("linux", "windows")},
                }
            ],
        }
        self.write("qualification/policy.json", policy)
        subprocess.run(["git", "add", "Cargo.toml", "bindings", "qualification/policy.json"], cwd=self.root, check=True)
        self.receipt_path = self.root / "receipt.json"
        self.receipt = {
            "schema": MODULE.SCHEMA,
            "project": "core",
            "repository": "GBeurier/nirs4all-core",
            "source_sha": "a" * 40,
            "execution": {"location": "local", "github_actions": False, "performance_policy": "strict"},
            "policy_sha256": MODULE.digest(self.root / MODULE.POLICY),
            "input_fingerprints": MODULE.runtime_inputs(self.root, "core"),
            "hosts": [],
            "runs": [],
        }
        for host in ("linux", "windows"):
            self.receipt["hosts"].append({"id": host, "os_family": host, "os": host, "arch": "x86_64", "hostname": "unit-fixture", "cpu_count": 2, "ram_bytes": 1000, "tools": {"python": "unit-fixture"}})
            run = {
                "id": "parity",
                "host": host,
                "command": ["unit-fixture-probe"],
                "cwd": str(self.root),
                "started_at": "2026-10-08T00:00:00Z",
                "finished_at": "2026-10-08T00:00:01Z",
                "exit_code": 0,
                "environment": {},
                "input_fingerprints": self.receipt["input_fingerprints"],
            }
            self.write(f"qualification/{host}-inputs.json", {"mode": "captured", "source_sha": self.receipt["source_sha"], "run_identity": {field: run.get(field) for field in MODULE.RUN_IDENTITY}, "input_fingerprints": run["input_fingerprints"]})
            run["input_evidence"] = self.descriptor(f"qualification/{host}-inputs.json")
            (self.root / f"qualification/{host}.log").write_text("Retained synthetic unit-test log; no product run claimed.\n")
            self.write(f"qualification/{host}-native-capture.json", {"artifact": artifact, "run_identity": {field: run.get(field) for field in MODULE.RUN_IDENTITY}, "source_sha": self.receipt["source_sha"], "log_sha256": MODULE.digest(self.root / f"qualification/{host}.log")})
            report = {
                **run,
                "summary": {"passed": 3, "failed": 0, "skipped": 0},
                "skips": [],
                "facts": {"hpo_trials": 45, "max_abs_error": 0},
                "provenance": {"tools": {"python": "unit-fixture"}, **{k: [{**artifact, "evidence": self.descriptor(f"qualification/{host}-native-capture.json")}] for k in ("source_artifacts", "dependency_origins")}},
            }
            self.write(f"qualification/{host}.json", report)
            run.update(log=self.descriptor(f"qualification/{host}.log"), report=self.descriptor(f"qualification/{host}.json"))
            self.receipt["runs"].append(run)

    def write(self, rel, value):
        (self.root / rel).write_text(json.dumps(value))

    def descriptor(self, rel):
        path = self.root / rel
        return {"path": rel, "bytes": path.stat().st_size, "sha256": MODULE.digest(path)}

    def verify(self):
        self.receipt_path.write_text(json.dumps(self.receipt))
        return MODULE.verify(self.root, "core", self.receipt_path)

    def mutate_report(self, transform):
        path = self.receipt["runs"][0]["report"]["path"]
        report = MODULE.load(self.root / path)
        transform(report)
        self.write(path, report)
        self.receipt["runs"][0]["report"] = self.descriptor(path)

    def test_retained_matching_two_host_fixture_is_accepted(self):
        self.assertEqual(self.verify()["verified_local_gates"], 2)

    def test_source_and_new_runtime_file_refuse_stale_receipt(self):
        for added in (False, True):
            with self.subTest(added=added):
                path = self.root / ("bindings/added.rs" if added else "bindings/runtime.rs")
                path.write_text("changed\n")
                if added:
                    subprocess.run(["git", "add", "bindings/added.rs"], cwd=self.root, check=True)
                with self.assertRaisesRegex(ValueError, "fingerprints"):
                    self.verify()

    def test_workflow_only_change_does_not_invalidate_runtime(self):
        (self.root / ".github/workflows").mkdir(parents=True)
        (self.root / ".github/workflows/build.yml").write_text("# CI-only metadata\n")
        subprocess.run(["git", "add", ".github"], cwd=self.root, check=True)
        self.verify()

    def test_policy_change_invalidates_preceding_qualification(self):
        path = self.root / MODULE.POLICY
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "policy digest"):
            self.verify()

    def test_arbitrary_command_and_relabelled_old_source_are_refused(self):
        for mutation in (lambda r: r.update(command=["echo", "fake success"]), lambda r: r.update(input_fingerprints={"bindings/runtime.rs": "c" * 64})):
            original = copy.deepcopy(self.receipt)
            mutation(self.receipt["runs"][0])
            with self.assertRaises(ValueError):
                self.verify()
            self.receipt = original

    def inherit_linux(self):
        original = copy.deepcopy(self.receipt["runs"][0])
        source_sha = self.receipt["source_sha"]
        self.write("qualification/historical.json", {"source_sha": source_sha, "input_fingerprints": original["input_fingerprints"], "run": original})
        self.write("qualification/continuity.json", {"input_fingerprints": original["input_fingerprints"], "historical_manifest_sha256": MODULE.digest(self.root / "qualification/historical.json"), "captured_inputs_sha256": original["input_evidence"]["sha256"], "historical_source_sha": source_sha, "current_source_sha": source_sha})
        source = {"mode": "inherited", "input_fingerprints": original["input_fingerprints"], "historical_manifest": self.descriptor("qualification/historical.json"), "continuity_report": self.descriptor("qualification/continuity.json")}
        self.write("qualification/linux-inherited-inputs.json", source)
        self.receipt["runs"][0]["input_evidence"] = self.descriptor("qualification/linux-inherited-inputs.json")
        self.write("qualification/linux-inherited.json", MODULE.load(self.root / original["report"]["path"]))
        self.receipt["runs"][0]["report"] = self.descriptor("qualification/linux-inherited.json")
        self.mutate_report(lambda r: r.update(input_evidence=self.receipt["runs"][0]["input_evidence"]))
        return source

    def test_inherited_proof_requires_both_historical_inputs_and_continuity(self):
        source = self.inherit_linux()
        self.verify()
        historical = MODULE.load(self.root / "qualification/historical.json")
        historical["input_fingerprints"] = {"Cargo.toml": "c" * 64}
        self.write("qualification/historical.json", historical)
        source["historical_manifest"] = self.descriptor("qualification/historical.json")
        self.write("qualification/linux-inherited-inputs.json", source)
        self.receipt["runs"][0]["input_evidence"] = self.descriptor("qualification/linux-inherited-inputs.json")
        self.mutate_report(lambda r: r.update(input_evidence=self.receipt["runs"][0]["input_evidence"]))
        with self.assertRaisesRegex(ValueError, "unchanged scoped inputs"):
            self.verify()

    def test_inherited_rejects_other_run_failed_exit_and_unbound_continuity(self):
        source = self.inherit_linux()
        original = MODULE.load(self.root / "qualification/historical.json")
        for mutate in (lambda h: h["run"].update(id="other-gate"), lambda h: h["run"].update(command=["echo"]), lambda h: h["run"].update(exit_code=19)):
            historical = copy.deepcopy(original)
            mutate(historical)
            self.write("qualification/historical.json", historical)
            source["historical_manifest"] = self.descriptor("qualification/historical.json")
            self.write("qualification/linux-inherited-inputs.json", source)
            self.receipt["runs"][0]["input_evidence"] = self.descriptor("qualification/linux-inherited-inputs.json")
            self.mutate_report(lambda r: r.update(input_evidence=self.receipt["runs"][0]["input_evidence"]))
            with self.assertRaises(ValueError):
                self.verify()
        self.write("qualification/historical.json", original)
        source["historical_manifest"] = self.descriptor("qualification/historical.json")
        continuity_original = MODULE.load(self.root / "qualification/continuity.json")
        for field in ("historical_manifest_sha256", "captured_inputs_sha256", "historical_source_sha", "current_source_sha"):
            continuity = copy.deepcopy(continuity_original)
            continuity[field] = "d" * 64
            self.write("qualification/continuity.json", continuity)
            source["continuity_report"] = self.descriptor("qualification/continuity.json")
            self.write("qualification/linux-inherited-inputs.json", source)
            self.receipt["runs"][0]["input_evidence"] = self.descriptor("qualification/linux-inherited-inputs.json")
            self.mutate_report(lambda r: r.update(input_evidence=self.receipt["runs"][0]["input_evidence"]))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify()

    def test_native_artifact_bytes_and_policy_cohort_are_bound(self):
        path = self.root / "qualification/native.bin"
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        with self.assertRaisesRegex(ValueError, "native artifact.*mismatch"):
            self.verify()
        path.write_bytes(original)
        self.mutate_report(lambda r: r["provenance"]["source_artifacts"][0].update(version="wrong-cohort"))
        with self.assertRaises(ValueError):
            self.verify()

    def test_portable_artifact_capture_survives_unmounted_origin(self):
        (self.root / "qualification/native.bin").unlink()
        self.verify()

    def test_fake_absolute_tool_homonym_is_refused(self):
        self.receipt["runs"][0]["command"] = ["/tmp/fake/unit-fixture-probe"]
        with self.assertRaisesRegex(ValueError, "executable path"):
            self.verify()

    def test_windows_root_path_suffix_normalizes_without_changing_other_literals(self):
        cwd = r"C:\owned\source"
        command = ["python.exe", cwd + r"\scripts\probe.py", "--config", cwd + r"\qualification\config.json", r"D:\external\literal.json", r"regex\n", r"{root}\already-literal"]
        self.assertEqual(MODULE.canonical_command(command, cwd, {}, "windows"), ["python", "{root}/scripts/probe.py", "--config", "{root}/qualification/config.json", r"D:\external\literal.json", r"regex\n", r"{root}\already-literal"])

    def test_bare_command_cannot_bypass_a_configured_absolute_executable(self):
        gate = {"executable_paths": {"linux": ["/usr/bin/python3"]}}
        with self.assertRaisesRegex(ValueError, "executable path"):
            MODULE.canonical_command(["python", "probe.py"], str(self.root), gate, "linux")
        self.assertEqual(MODULE.canonical_command(["/usr/bin/python3", "probe.py"], str(self.root), gate, "linux"), ["python", "probe.py"])

    def test_empty_authentic_logs_are_allowed_but_other_evidence_remains_nonempty(self):
        (self.root / "qualification/empty.log").write_bytes(b"")
        row = self.descriptor("qualification/empty.log")
        MODULE.evidence(self.root, row, "log")
        MODULE.evidence(self.root, row, "historical log")
        for label in ("report", "input evidence", "native artifact capture"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                MODULE.evidence(self.root, row, label)

    def controlled_history(self):
        run = self.receipt["runs"][0]
        run.update(started_at=None, finished_at=None, timing_basis="historical-duration", duration_seconds=0.25)
        direct = {"command": run["command"], "direct_command_exit_code": 0, "seconds": 0.25, "independent_child_wrapper_pid": 1}
        raw = {"gate": run["id"], "command": run["command"], "exit_code": 0, "stop_reason": None, "functional_source_unchanged": True,
               "independent_command_terminal": direct, "logs": {"stdout.log": {field: run["log"][field] for field in ("bytes", "sha256")}}}
        originals = {"raw_receipt": raw, "direct_terminal": direct, "active": {"command": run["command"], "worktree": run["cwd"]},
                     "source_freeze": {"virtual_full_git_tree": self.receipt["source_sha"]}}
        for name, value in originals.items():
            self.write(f"qualification/original-{name}.json", value)
        for name in ("runner", "child_wrapper"):
            (self.root / f"qualification/original-{name}.py").write_text("# Synthetic controlled-runner fixture; not product evidence\n")
        descriptors = {name: self.descriptor(f"qualification/original-{name}.json") for name in originals}
        descriptors.update({name: self.descriptor(f"qualification/original-{name}.py") for name in ("runner", "child_wrapper")})
        historical = {"format": "controlled-runner-duration.v1", "source_sha": self.receipt["source_sha"], "source_basis": "reconstructed_from_reviewed_tree",
                      "input_fingerprints": run["input_fingerprints"], "evidence": descriptors, "retained_logs": {"stdout.log": run["log"]}}
        self.write("qualification/controlled-historical.json", historical)
        self.write("qualification/controlled-continuity.json", {"historical_manifest_sha256": MODULE.digest(self.root / "qualification/controlled-historical.json"),
                  "historical_source_sha": self.receipt["source_sha"], "current_source_sha": self.receipt["source_sha"], "input_fingerprints": run["input_fingerprints"]})
        self.write("qualification/controlled-inputs.json", {"mode": "inherited", "lineage_format": "controlled-runner-duration.v1", "input_fingerprints": run["input_fingerprints"],
                  "historical_manifest": self.descriptor("qualification/controlled-historical.json"), "continuity_report": self.descriptor("qualification/controlled-continuity.json")})
        run["input_evidence"] = self.descriptor("qualification/controlled-inputs.json")
        report = MODULE.load(self.root / run["report"]["path"])
        report.update({field: run.get(field) for field in (*MODULE.RUN_IDENTITY, "input_evidence", "timing_basis", "duration_seconds")})
        for field in ("source_artifacts", "dependency_origins"):
            for row in report["provenance"][field]:
                path = row["evidence"]["path"]
                native = MODULE.load(self.root / path)
                native["run_identity"] = {field: run.get(field) for field in MODULE.RUN_IDENTITY}
                self.write(path, native)
                row["evidence"] = self.descriptor(path)
        self.write(run["report"]["path"], report)
        run["report"] = self.descriptor(run["report"]["path"])
        authority = {"format": historical["format"], "source_sha": historical["source_sha"], "source_basis": historical["source_basis"],
                     "evidence_sha256": {name: row["sha256"] for name, row in descriptors.items()}, "scoped_inputs_sha256": MODULE.json_digest(run["input_fingerprints"]),
                     "source_tree_field": "virtual_full_git_tree", "source_verified_field": "functional_source_unchanged", "environment_basis": "derived_from_retained_controlled_runner",
                     "environment": {}, "primary_log": "stdout.log", "observations_sha256": MODULE.json_digest({field: report.get(field) for field in ("summary", "skips", "facts", "provenance", "raw_reports")})}
        policy = MODULE.load(self.root / MODULE.POLICY)
        policy["gates"][0].update(historical_authority={"linux": authority}, executable_paths={"linux": ["/usr/bin/unit-fixture-probe"]})
        self.write(MODULE.POLICY, policy)
        self.receipt["policy_sha256"] = MODULE.digest(self.root / MODULE.POLICY)
        return authority

    def test_controlled_history_preserves_duration_without_fabricating_wall_timestamps(self):
        self.controlled_history()
        self.verify()
        self.receipt["runs"][0]["started_at"] = "2026-10-08T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "wall timestamps"):
            self.verify()

    def test_controlled_history_requires_exact_raw_authority_source_and_observations(self):
        self.controlled_history()
        self.verify()
        path = self.root / "qualification/original-direct_terminal.json"
        value = MODULE.load(path)
        value["direct_command_exit_code"] = 19
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.verify()

    def test_controlled_history_cannot_relabel_gate_duration_or_observations(self):
        self.controlled_history()
        original = copy.deepcopy(self.receipt)
        for change in (lambda r: r.update(id="other-gate"), lambda r: r.update(duration_seconds=99), lambda r: r.update(environment={"FAKE": "1"})):
            self.receipt = copy.deepcopy(original)
            change(self.receipt["runs"][0])
            with self.assertRaises(ValueError):
                self.verify()
        self.receipt = original
        self.mutate_report(lambda r: r["facts"].update(hpo_trials=99))
        with self.assertRaisesRegex(ValueError, "observations"):
            self.verify()

    def test_native_capture_relabel_and_updated_fake_hash_are_refused(self):
        path = "qualification/linux-native-capture.json"
        original = MODULE.load(self.root / path)
        for change in (lambda c: c["run_identity"].update(command=["echo"]), lambda c: c.update(source_sha="e" * 40), lambda c: c.update(log_sha256="e" * 64)):
            capture = copy.deepcopy(original)
            change(capture)
            self.write(path, capture)
            self.mutate_report(lambda r: [row.update(evidence=self.descriptor(path)) for field in ("source_artifacts", "dependency_origins") for row in r["provenance"][field]])
            with self.assertRaises(ValueError):
                self.verify()
        capture = copy.deepcopy(original)
        capture["artifact"]["sha256"] = "f" * 64
        self.write(path, capture)
        self.mutate_report(lambda r: [row.update(sha256="f" * 64, evidence=self.descriptor(path)) for field in ("source_artifacts", "dependency_origins") for row in r["provenance"][field]])
        with self.assertRaisesRegex(ValueError, "publication cohort"):
            self.verify()

    def test_sdk_scope_includes_runtime_examples_and_test_requirements(self):
        (self.root / "nirs4all").mkdir()
        (self.root / "examples").mkdir()
        for path in ("nirs4all/runtime.py", "examples/pipeline.py", "requirements-test.txt", "requirements-examples.txt"):
            (self.root / path).write_text("# SDK input fixture\n")
        subprocess.run(["git", "add", "nirs4all", "examples", "requirements-test.txt", "requirements-examples.txt"], cwd=self.root, check=True)
        selected = MODULE.runtime_inputs(self.root, "sdk")
        self.assertTrue({"nirs4all/runtime.py", "examples/pipeline.py", "requirements-test.txt", "requirements-examples.txt"} <= selected.keys())

    def test_tracked_workflow_and_docs_can_be_real_gate_inputs(self):
        (self.root / ".github/workflows").mkdir(parents=True)
        (self.root / ".github/workflows/build.yml").write_text("# actual validator input\n")
        (self.root / "README.md").write_text("Actual public-docs validator input\n")
        subprocess.run(["git", "add", ".github", "README.md"], cwd=self.root, check=True)
        policy = MODULE.load(self.root / MODULE.POLICY)
        policy["gates"][0]["input_paths"] += [".github/workflows/build.yml", "README.md"]
        self.write(MODULE.POLICY, policy)
        self.receipt["policy_sha256"] = MODULE.digest(self.root / MODULE.POLICY)
        selected = MODULE.gate_inputs(MODULE.tracked_inputs(self.root), policy["gates"][0])
        for run in self.receipt["runs"]:
            run["input_fingerprints"] = selected
            capture = MODULE.load(self.root / run["input_evidence"]["path"])
            capture.update(input_fingerprints=selected, run_identity={field: run.get(field) for field in MODULE.RUN_IDENTITY})
            self.write(run["input_evidence"]["path"], capture)
            run["input_evidence"] = self.descriptor(run["input_evidence"]["path"])
            report = MODULE.load(self.root / run["report"]["path"])
            report.update(input_fingerprints=selected, input_evidence=run["input_evidence"])
            for rows in report["provenance"].values():
                if isinstance(rows, list):
                    for row in rows:
                        capture_path = row["evidence"]["path"]
                        native_capture = MODULE.load(self.root / capture_path)
                        native_capture["run_identity"] = {field: run.get(field) for field in MODULE.RUN_IDENTITY}
                        self.write(capture_path, native_capture)
                        row["evidence"] = self.descriptor(capture_path)
            self.write(run["report"]["path"], report)
            run["report"] = self.descriptor(run["report"]["path"])
        self.verify()
        (self.root / "README.md").write_text("changed validator input\n")
        with self.assertRaisesRegex(ValueError, "run source fingerprint"):
            self.verify()

    def test_scoped_gate_can_inherit_when_unrelated_runtime_inputs_change(self):
        policy = MODULE.load(self.root / MODULE.POLICY)
        policy["gates"][0]["input_paths"] = ["Cargo.toml", "bindings/runtime.rs"]
        self.write(MODULE.POLICY, policy)
        self.receipt["policy_sha256"] = MODULE.digest(self.root / MODULE.POLICY)
        (self.root / "bindings/unrelated.rs").write_text("// unrelated to this scoped gate\n")
        subprocess.run(["git", "add", "bindings/unrelated.rs"], cwd=self.root, check=True)
        self.receipt["input_fingerprints"] = MODULE.runtime_inputs(self.root, "core")
        self.verify()

    def test_incomplete_or_one_host_policy_and_missing_scoped_input_are_refused(self):
        original = MODULE.load(self.root / MODULE.POLICY)
        for change in (lambda p: p.update(proposal_incomplete_not_for_publication=True), lambda p: p["gates"][0].update(hosts=["linux"]), lambda p: p["gates"][0]["input_paths"].append("bindings/absent.rs")):
            policy = copy.deepcopy(original)
            change(policy)
            self.write(MODULE.POLICY, policy)
            self.receipt["policy_sha256"] = MODULE.digest(self.root / MODULE.POLICY)
            with self.assertRaises(ValueError):
                self.verify()

    def test_cv_vector_requires_nonempty_finite_actual_numeric_values(self):
        rule = {"field": "cv_scores", "op": "finite_nonempty"}
        MODULE.fact_check({"cv_scores": [0.1, 0.2]}, rule)
        for values in ([], [True], [float("nan")], [float("inf")], ["0.1"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                MODULE.fact_check({"cv_scores": values}, rule)

    def test_playwright_checks_actual_stats_errors_and_individual_results(self):
        raw = {
            "stats": {"expected": 3, "unexpected": 0, "skipped": 0, "flaky": 0},
            "errors": [],
            "suites": [{"specs": [{"tests": [{"expectedStatus": "passed", "status": "expected", "results": [{"status": "passed"}]} for _ in range(3)]}]}],
        }
        self.write("qualification/playwright.json", raw)
        report = {"raw_reports": {"playwright": self.descriptor("qualification/playwright.json")}}
        MODULE.check_playwright_report(self.root, report, {"minimum_passed": 3})
        for mutate in (
            lambda r: r.update(suites=[]),
            lambda r: r.update(errors=[{"message": "actual failure"}]),
            lambda r: r["stats"].update(skipped=1),
            lambda r: r["suites"][0]["specs"][0]["tests"][0]["results"][0].update(status="failed"),
        ):
            broken = copy.deepcopy(raw)
            mutate(broken)
            self.write("qualification/playwright.json", broken)
            report["raw_reports"]["playwright"] = self.descriptor("qualification/playwright.json")
            with self.assertRaises(ValueError):
                MODULE.check_playwright_report(self.root, report, {"minimum_passed": 3})

    def test_missing_host_or_run_and_remote_origin_are_refused(self):
        original = copy.deepcopy(self.receipt)
        for mutation in (lambda r: r["hosts"].pop(), lambda r: r["runs"].pop(), lambda r: r["execution"].update(github_actions=True)):
            with self.subTest(mutation=mutation):
                self.receipt = copy.deepcopy(original)
                mutation(self.receipt)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_negative_facts_failures_and_unapproved_skips_are_refused(self):
        changes = (
            lambda r: r["facts"].update(hpo_trials=44),
            lambda r: r["facts"].update(max_abs_error=1),
            lambda r: r["summary"].update(failed=1),
            lambda r: (r["summary"].update(skipped=1), r.update(skips=["unapproved"])),
            lambda r: r.update(exit_code=1),
            lambda r: r.update(provenance={}),
        )
        original_receipt = copy.deepcopy(self.receipt)
        original_report = MODULE.load(self.root / self.receipt["runs"][0]["report"]["path"])
        for change in changes:
            with self.subTest(change=change):
                self.receipt = copy.deepcopy(original_receipt)
                self.write("qualification/linux.json", original_report)
                self.mutate_report(change)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_boolean_or_nan_cannot_substitute_measured_numeric_fact(self):
        for value in (True, float("nan")):
            with self.subTest(value=value):
                self.mutate_report(lambda r, selected=value: r["facts"].update(hpo_trials=selected))
                with self.assertRaisesRegex(ValueError, "finite numeric"):
                    self.verify()

    def test_changed_log_or_report_bytes_are_refused(self):
        for field in ("log", "report"):
            with self.subTest(field=field):
                path = self.root / self.receipt["runs"][0][field]["path"]
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    self.verify()
                path.write_bytes(original)

    def test_duplicate_json_key_and_path_escape_are_refused(self):
        path = self.root / "ambiguous.json"
        path.write_text('{"schema":1,"schema":2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.load(path)
        for rel in ("../secret", "/secret", "qualification/../../secret", "C:/secret"):
            with self.subTest(rel=rel), self.assertRaises(ValueError):
                MODULE.relative_file(self.root, rel)


if __name__ == "__main__":
    unittest.main()
