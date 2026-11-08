"""Isolated source/envelope contract tests; no SDK or estimator imports."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

# The isolated harness may inject original/mutant modules. Default collection
# loads only these two stdlib-only files, without importing the SDK package.
if "foundation_modules" in sys.modules:
    identity = sys.modules["foundation_modules"].identity
    source = sys.modules["foundation_modules"].source
else:
    directory = Path(__file__).resolve().parents[4] / "nirs4all/pipeline/config"
    package = types.ModuleType("identity_test_source")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    for module_name in ("variant_identity", "recipe_source"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{module_name}", directory / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    identity = sys.modules["identity_test_source.variant_identity"]
    source = sys.modules["identity_test_source.recipe_source"]


# This child is deliberately self-contained: ordinary unit collection does not
# depend on a private harness file. Only stdlib and the selected two source files
# are loaded. The deadline is a hang backstop, not a speed assertion. The
# communicate API is portable; guarded Windows execution is not qualified.
_BOUNDED_PROBE = r'''
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import types

assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
protected = ("HOME", "home", "CODEX_HOME")
threads = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
assert all(os.environ.get(k) == "1" for k in threads)
actual = hashlib.sha256(json.dumps([os.environ.get(k) for k in protected]).encode()).hexdigest()
assert actual == os.environ["IDENTITY_PROBE_PARENT_ENV_SHA256"]
forbidden = {"nirs4all", "numpy", "scipy", "sklearn", "polars", "dag_ml", "dag_ml_data", "n4m", "pls4all", "torch", "tensorflow", "jax"}
compute = {"fit", "partial_fit", "fit_transform", "fit_predict", "transform", "predict", "predict_proba", "run_native_methods"}
events = {"blocked": 0, "projector": 0}
work = {"encode_recipe_calls": 0}
encode_code = None

class GuardRefusal(BaseException):
    pass

class Imports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in forbidden:
            events["blocked"] += 1
            raise GuardRefusal("product/scientific import refused")
        return None

def profile(frame, event, arg):
    # Count entry to the actual function body, without wrapping or replacing
    # any candidate function. This is independent of clock speed.
    if event == "call" and frame.f_code is encode_code:
        work["encode_recipe_calls"] += 1
    if event == "call" and frame.f_code.co_name in compute:
        events["blocked"] += 1
        raise GuardRefusal("scientific runtime operation refused")

def deny_thread(*args, **kwargs):
    events["blocked"] += 1
    raise GuardRefusal("worker creation refused")

def audit(event, args):
    if (event in {"sqlite3.connect", "ctypes.dlopen", "subprocess.Popen", "os.system", "os.fork", "os.forkpty", "os.posix_spawn", "pty.spawn"}
            or event.startswith(("socket.", "os.exec", "os.spawn"))):
        events["blocked"] += 1
        raise GuardRefusal("external/native operation refused")

assert not any(n.split(".")[0] in forbidden for n in sys.modules)
sys.meta_path.insert(0, Imports())
sys.setprofile(profile)
threading.setprofile(profile)
threading.Thread.start = deny_thread
sys.addaudithook(audit)
assert sys.getprofile() is profile and threading.getprofile() is profile
# This qualification runs on Linux. POSIX gets an address-space ceiling.
# The Windows branch is unexecuted; a parent Thread.start tripwire can also
# conflict with Windows communicate reader threads. No universal guard claim.
memory_limit = None
if os.name == "posix":
    import resource
    memory_limit = 256 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

directory, operation, kind = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
package = types.ModuleType("bounded_identity_source")
package.__path__ = [str(directory)]
sys.modules[package.__name__] = package
mods = {}
pins = {}
for name in ("variant_identity", "recipe_source"):
    path = directory / (name + ".py")
    assert not path.is_symlink()
    pins[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(package.__name__ + "." + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    mods[name] = module
identity, source = mods["variant_identity"], mods["recipe_source"]
encode_code = identity.encode_recipe.__code__
node = {"x": (1, 2.0)} if kind == "conversion" else "x"
for _ in range(40):
    node = {"a": node, "b": node} if kind == "conversion" else [node, node]
value = {"finetune_params": node} if kind == "conversion" else node
if operation == "constructor" and kind != "conversion":
    value = {"cfg": value}

def projector(parameters):
    events["projector"] += 1
    print(json.dumps({"phase": "projector_called"}), flush=True)
    return {}

print(json.dumps({"phase": "operation_started", "pid": os.getpid(), "guard_before_source": True,
                  "protected_inheritance_verified": True, "six_thread_limits_one": True,
                  "threading_start_replaced": True, "thread_profile_installed": True,
                  "low_level_thread_creation": "not universally intercepted",
                  "memory_limit_bytes": memory_limit, "source_sha256": pins}), flush=True)
try:
    if operation == "encode":
        identity.encode_recipe(value)
    elif operation == "capture":
        source.capture_source(value)
    elif operation == "constructor":
        source.project_constructor(value, projector)
    else:
        raise AssertionError("unknown fixture operation")
except identity.IdentityError as error:
    result = {"status": "refused", "code": error.code}
else:
    result = {"status": "accepted"}
assert not any(n.split(".")[0] in forbidden for n in sys.modules)
print(json.dumps({"phase": "complete", **result, "events": events, "work": work}), flush=True)
'''


def _bounded_probe(operation, kind):
    """Run exactly the selected source under an inherited-env, guarded deadline."""
    protected = ("HOME", "home", "CODEX_HOME")
    threads = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    overrides = dict.fromkeys(threads, "1")
    overrides["IDENTITY_PROBE_PARENT_ENV_SHA256"] = hashlib.sha256(json.dumps([os.environ.get(k) for k in protected]).encode()).hexdigest()
    assert not set(protected).intersection(overrides)
    childenv = os.environ.copy()
    childenv.update(overrides)
    equality = {key: childenv.get(key) == os.environ.get(key) for key in protected}
    assert all(equality.values())
    assert all(childenv[key] == "1" for key in threads)
    argv = [sys.executable, "-I", "-S", "-B", "-c", _BOUNDED_PROBE, str(Path(identity.__file__).parent), operation, kind]
    guard = sys.modules.get("foundation_guard")
    permission = guard.permit_probe(argv, childenv) if guard is not None and hasattr(guard, "permit_probe") else nullcontext()
    started = time.monotonic()
    timed_out = False
    with permission:
        # Assert immediately before EACH launch, including probes nested in tests.
        assert not set(protected).intersection(overrides)
        assert all(childenv.get(key) == os.environ.get(key) for key in protected)
        process = subprocess.Popen(argv, executable=sys.executable, env=childenv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate(timeout=10)
    messages = [json.loads(line) for line in stdout.splitlines()]
    record = {"probe_child": True, "pid": process.pid, "parent_pid": os.getpid(), "argv": argv, "exit": process.returncode,
              "timeout": timed_out, "timeout_seconds": 120, "seconds": time.monotonic() - started,
              "protected_equal": equality, "protected_override_absent": True, "six_thread_limits_one": True,
              "messages": messages, "stderr": stderr}
    print(json.dumps(record), flush=True)
    return record


class TestProductionRegressions(unittest.TestCase):
    """Same assertions against exact old production bodies and new codec."""

    def test_plain_json_positive(self):
        value = {"pipeline": [{"model": {"class": "fixture.Model", "params": {"scale": False}}}], "steps": [], "provenance": {"version": "1.2"}}
        self.assertEqual(json.loads(identity.encode_recipe(value).payload), value)

    def test_metadata_positive_and_input_not_mutated(self):
        value = {"name": "fixture.module.label", "metadata": {"source": "numpy.linalg.norm", "order": [3, 1, 2]}}
        before = json.dumps(value)
        identity.encode_recipe(value)
        self.assertEqual(json.dumps(value), before)

    def test_tuple_refused(self):
        with self.assertRaises(ValueError):
            identity.encode_recipe({"params": {"pair": (1, 2)}})

    def test_boolean_mapping_key_refused(self):
        with self.assertRaises(ValueError):
            identity.encode_recipe({"steps": {True: ["yes"], False: ["no"]}})

    def test_coercing_collision_refused(self):
        with self.assertRaises(ValueError):
            identity.encode_recipe({True: 1, "true": 2})

    def test_nan_refused(self):
        with self.assertRaises(ValueError):
            identity.encode_recipe({"params": {"value": float("nan")}})

    def test_infinity_refused(self):
        with self.assertRaises(ValueError):
            identity.encode_recipe({"params": {"value": float("inf")}})

    def test_no_default_stringification(self):
        calls = []

        class Unknown:
            def __str__(self):
                calls.append("stringified")
                return "lost constructor state"

        with self.assertRaises(ValueError):
            identity.encode_recipe({"operator": Unknown()})
        self.assertEqual(calls, [])

    def test_mapping_order_is_identity(self):
        self.assertNotEqual(identity.encode_recipe({"a": 1, "b": 2}).digest, identity.encode_recipe({"b": 2, "a": 1}).digest)


class TestRecipeCodec(unittest.TestCase):
    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(identity.IdentityError) as raised:
            function(*args, **kwargs)
        self.assertEqual(raised.exception.code, code)
        return raised.exception

    def test_typed_numbers_and_negative_zero(self):
        values = [True, 1, 1.0, 0.0, -0.0]
        self.assertEqual(len({identity.encode_recipe(v).digest for v in values}), len(values))

    def test_unicode_not_normalized(self):
        self.assertNotEqual(identity.encode_recipe("\u00e9").digest, identity.encode_recipe("e\u0301").digest)

    def test_surrogate_refused(self):
        self.assert_code("invalid_unicode", identity.encode_recipe, {"name": "\ud800"})

    def test_nested_error_location(self):
        error = self.assert_code("tuple_not_supported", identity.encode_recipe, {"params": {"items": [{"pair": (1, 2)}]}})
        self.assertEqual(error.path, ("params", "items", 0, "pair"))

    def test_roundtrip_and_independent_copies(self):
        value = {"pipeline": [{"params": {"scale": False, "zero": -0.0}}], "description": "fixture"}
        encoded = identity.encode_recipe(value)
        decoded = identity.verify_recipe(encoded.payload, encoded.digest)
        self.assertEqual(identity.encode_recipe(decoded).canonical, encoded.canonical)
        decoded["pipeline"].append("changed")
        self.assertEqual(encoded.value(), value)
        value["pipeline"].append("caller changed")
        self.assertNotEqual(encoded.value(), value)

    def test_payload_tamper(self):
        encoded = identity.encode_recipe({"scale": False})
        self.assert_code("digest_mismatch", identity.verify_recipe, encoded.payload.replace(b"false", b"true"), encoded.digest)

    def test_noncanonical_stored_bytes_refused(self):
        encoded = identity.encode_recipe({"x": 1})
        self.assert_code("noncanonical_payload", identity.verify_recipe, b'{"x": 1}', encoded.digest)

    def test_wrong_profile_and_digest(self):
        encoded = identity.encode_recipe({})
        self.assert_code("profile_unsupported", identity.verify_recipe, encoded.payload, encoded.digest, profile="other.v1")
        self.assert_code("digest_invalid", identity.verify_recipe, encoded.payload, "not-a-digest")

    def test_duplicate_json_keys(self):
        self.assert_code("duplicate_key", identity.decode_json, b'{"x":1,"x":2}')

    def test_decoder_nonfinite_and_overflow(self):
        for payload in (b'NaN', b'Infinity', b'-Infinity', b'1e999'):
            with self.subTest(payload=payload):
                self.assert_code("nonfinite_number", identity.decode_json, payload)

    def test_decoder_invalid_utf8_and_json(self):
        self.assert_code("invalid_unicode", identity.decode_json, b'"\xff"')
        self.assert_code("invalid_json", identity.decode_json, b'{')

    def test_cycles_but_shared_noncyclic_values_valid(self):
        value = []
        value.append(value)
        self.assert_code("cycle", identity.encode_recipe, value)
        shared = [1]
        self.assertEqual(identity.encode_recipe([shared, shared]).value(), [[1], [1]])

    def test_caps_and_depth(self):
        self.assert_code("size_limit", identity.encode_recipe, "x" * 200, max_bytes=80)
        self.assert_code("depth_limit", identity.encode_recipe, [[[0]]], max_depth=2)
        self.assert_code("size_limit", identity.decode_json, b'"' + b'x' * 100 + b'"', max_bytes=80)
        self.assert_code("limit_invalid", identity.encode_recipe, {}, max_depth=129)

    def test_subclasses_not_silently_coerced(self):
        class NamedInt(int):
            pass

        self.assert_code("value_not_supported", identity.encode_recipe, NamedInt(1))


class TestSourceEnvelope(unittest.TestCase):
    assert_code = TestRecipeCodec.assert_code

    def test_root_shapes_and_all_keys(self):
        for document in ([{"model": "fixture.Model"}], {"pipeline": [], "steps": ["kept"], "name": "fixture.name", "random_state": 7, "metadata": {"x": [3, 1, 2]}}):
            with self.subTest(document=document):
                captured = source.capture_source(document)
                self.assertEqual(captured.document(), document)
                self.assertEqual(source.read_source(captured.payload, captured.digest).document(), document)
                self.assertIsNone(captured.original_bytes())

    def test_exact_raw_json_bytes_preserved(self):
        raw = b'{\n "steps" : [], "name":"literal.module.label"\n}\n'
        captured = source.capture_json_source(raw)
        self.assertEqual(captured.original_bytes(), raw)
        loaded = source.read_source(captured.payload, captured.digest)
        self.assertEqual(loaded.original_bytes(), raw)
        self.assertNotEqual(captured.digest, source.capture_json_source(b'{"steps":[],"name":"literal.module.label"}').digest)

    def test_source_copies_no_alias(self):
        document = {"steps": [{"metadata": {"order": [3, 1, 2]}}]}
        captured = source.capture_source(document)
        captured.document()["steps"].clear()
        captured.normalized_document()["steps"].clear()
        self.assertEqual(captured.document(), document)
        document["steps"].clear()
        self.assertNotEqual(captured.document(), document)

    def test_declared_two_tuple_finetune_conversion(self):
        document = {"steps": [{"finetune_params": {"model_params": {"n_components": (1, 30), "alpha": (0.1, 0.5)}}}]}
        captured = source.capture_source(document)
        self.assertEqual(captured.document(), document)
        params = captured.normalized_document()["steps"][0]["finetune_params"]["model_params"]
        self.assertEqual(params["n_components"], {"type": "int", "min": 1, "max": 30})
        self.assertEqual(params["alpha"], {"type": "float", "min": 0.1, "max": 0.5})
        self.assertEqual(len(captured.conversions()), 2)
        self.assertEqual(source.read_source(captured.payload, captured.digest).document(), document)

    def test_tuple_categorical_and_documented_three_tuple_refuse(self):
        for value in (("int", 1, 30), [(1, 2)], (True, 2)):
            with self.subTest(value=value):
                self.assert_code("tuple_not_supported", source.capture_source, {"finetune_params": {"x": value}})

    def test_ordinary_pair_and_nonstring_keys_refuse(self):
        self.assert_code("tuple_not_supported", source.capture_source, {"params": {"pair": (1, 2)}})
        self.assert_code("key_not_supported", source.capture_source, {"branch": {"steps": {True: []}}})

    def test_constructor_precheck_before_projection(self):
        calls = []

        def projector(parameters):
            calls.append("projected")
            return {"class": "fixture.Dummy", "params": {"pair": list(parameters["pair"])}}

        self.assert_code("tuple_not_supported", source.project_constructor, {"pair": (1, 2)}, projector)
        self.assertEqual(calls, [])

    def test_constructor_projection_explicit_positive(self):
        calls = []

        def projector(parameters):
            calls.append("projected")
            parameters["cfg"]["name"] = "fixture.callable"
            return {"class": "fixture.Dummy", "params": parameters}

        parameters = {"cfg": {"name": "descriptive.original"}, "scale": False}
        result = source.project_constructor(parameters, projector)
        self.assertEqual(calls, ["projected"])
        self.assertEqual(result.source.document(), parameters)
        self.assertEqual(result.recipe.value()["params"]["cfg"]["name"], "fixture.callable")
        self.assertEqual(parameters["cfg"]["name"], "descriptive.original")

    def test_unknown_object_does_not_import_or_construct(self):
        class Live:
            def __str__(self):
                self.fail = "must not stringify"
                raise AssertionError("must not stringify")

        self.assert_code("value_not_supported", source.capture_source, {"model": Live()})
        self.assert_code("constructor_parameters_unavailable", source.project_constructor, None, lambda _: {})

    def test_source_tamper_wrong_schema_and_fields(self):
        captured = source.capture_source({"steps": []})
        self.assert_code("digest_mismatch", source.read_source, captured.payload, "sha256:" + "0" * 64)
        value = json.loads(captured.payload)
        for key, replacement in (("schema", "other"), ("representation", "guess"), ("conversions", [{"path": ["missing"], "original": [1, 2]}])):
            changed = dict(value)
            changed[key] = replacement
            payload = identity.encode_recipe(changed).payload
            import hashlib
            with self.assertRaises(identity.IdentityError):
                source.read_source(payload, "sha256:" + hashlib.sha256(payload).hexdigest())

    def test_nonfinite_range_and_source_size(self):
        self.assert_code("nonfinite_number", source.capture_source, {"finetune_params": {"x": (1, float("inf"))}})
        self.assert_code("size_limit", source.capture_source, {"steps": ["x" * 200]}, max_bytes=100)
        self.assert_code("source_root_invalid", source.capture_source, "bare label")

    def test_fake_storage_uses_exact_validated_bytes(self):
        encoded = identity.encode_recipe({"params": {"scale": False}})
        rows = []
        rows.append(encoded.payload)
        self.assertIs(rows[0], encoded.payload)
        self.assertEqual(identity.verify_recipe(rows[0], encoded.digest), {"params": {"scale": False}})

    def test_raw_source_cannot_disagree_with_document(self):
        import hashlib

        captured = source.capture_json_source(b'{"steps":[]}')
        changed = json.loads(captured.payload)
        changed["document"]["steps"] = ["changed"]
        payload = identity.encode_recipe(changed).payload
        self.assert_code("source_document_mismatch", source.read_source, payload, "sha256:" + hashlib.sha256(payload).hexdigest())

    def test_conversion_cannot_hide_ordinary_tuple(self):
        import hashlib

        captured = source.capture_source({"params": {"pair": {"type": "int", "min": 1, "max": 2}}})
        changed = json.loads(captured.payload)
        changed["conversions"] = [{"path": ["params", "pair"], "original": [1, 2]}]
        payload = identity.encode_recipe(changed).payload
        self.assert_code("tuple_not_supported", source.read_source, payload, "sha256:" + hashlib.sha256(payload).hexdigest())

    def test_explicit_fake_constructor_exposes_parameters_once(self):
        events = []

        class Dummy:
            def __init__(self, *, scale):
                events.append("constructor")
                self.scale = scale

            def get_params(self):
                events.append("parameters")
                return {"scale": self.scale}

        dummy = Dummy(scale=False)
        result = source.project_constructor(dummy.get_params(), lambda params: {"class": "fixture.Dummy", "params": params})
        self.assertEqual(events, ["constructor", "parameters"])
        self.assertEqual(result.recipe.value(), {"class": "fixture.Dummy", "params": {"scale": False}})


class TestWorkCorrections(unittest.TestCase):
    """Concrete reviewer counterexamples and independent contract pins."""

    assert_code = TestRecipeCodec.assert_code

    @staticmethod
    def dag(levels, leaf="x"):
        for _ in range(levels):
            leaf = [leaf, leaf]
        return leaf

    def bounded_refusal(self, operation, kind="plain"):
        result = _bounded_probe(operation, kind)
        self.assertTrue(result["messages"], result["stderr"])
        ready = result["messages"][0]
        self.assertEqual(ready["phase"], "operation_started")
        self.assertTrue(ready["guard_before_source"])
        self.assertTrue(ready["protected_inheritance_verified"])
        self.assertTrue(ready["six_thread_limits_one"])
        self.assertFalse(result["timeout"], "probe hit the generous hang backstop; not a semantic mutant kill")
        self.assertEqual(result["exit"], 0, result["stderr"])
        complete = result["messages"][-1]
        self.assertEqual(complete["phase"], "complete")
        self.assertEqual(complete["status"], "refused")
        self.assertEqual(complete["code"], "size_limit")
        self.assertEqual(complete["events"], {"blocked": 0, "projector": 0})
        self.assertGreater(complete["work"]["encode_recipe_calls"], 0)
        if kind == "conversion":
            # Literal upper bound independently measured on this exact fixture
            # by the retained reviewer, not derived from the loaded source or
            # generated from a mutant. Full/half/quarter charge faults must all
            # finish normally and fail HERE on excess actual function entries.
            self.assertLessEqual(complete["work"]["encode_recipe_calls"], 2_342, "conversion DAG exceeded the encode_recipe work bound")

    def test_dag40_encode_bounded(self):
        self.bounded_refusal("encode")

    def test_dag40_capture_bounded(self):
        self.bounded_refusal("capture")

    def test_dag40_constructor_preflight_bounded(self):
        self.bounded_refusal("constructor")

    def test_conversion_dag40_capture_bounded(self):
        self.bounded_refusal("capture", "conversion")

    def test_conversion_dag40_constructor_preflight_bounded(self):
        self.bounded_refusal("constructor", "conversion")

    def test_conversion_dag_positive_preserves_every_occurrence(self):
        node = {"x": (1, 2.0)}
        for _ in range(3):
            node = {"a": node, "b": node}
        value = {"finetune_params": node}
        captured = source.capture_source(value)
        self.assertEqual(captured.document(), value)
        self.assertEqual(len(captured.conversions()), 8)
        self.assertEqual(len({tuple(c["path"]) for c in captured.conversions()}), 8)
        for conversion in captured.conversions():
            current = captured.document()
            for key in conversion["path"]:
                current = current[key]
            self.assertIs(type(current), tuple)
            self.assertEqual([type(v) for v in current], [int, float])
        out = captured.document()
        out["finetune_params"]["a"]["a"]["a"]["x"] = "changed"
        self.assertEqual(out["finetune_params"]["b"]["a"]["a"]["x"], (1, 2.0))
        self.assertEqual(captured.document(), value)

    def test_shared_valid_graphs_are_copied_not_truncated(self):
        shared = self.dag(5, {"name": "literal.name", "scale": False})
        value = [shared, {"nested": shared}, shared]
        encoded = identity.encode_recipe(value)
        self.assertEqual(encoded.value(), value)
        self.assertEqual(encoded.payload, identity.encode_recipe(json.loads(encoded.payload)).payload)
        captured = source.capture_source(value)
        self.assertEqual(captured.document(), value)
        out = captured.document()
        out[0].clear()
        self.assertEqual(out[1]["nested"], shared)
        self.assertEqual(out[2], shared)
        self.assertEqual(captured.document(), value)

    def test_shared_ranges_keep_every_located_conversion(self):
        shared = {"x": (1, 2.0)}
        value = {"finetune_params": {"a": shared, "b": shared}}
        captured = source.capture_source(value)
        self.assertEqual(captured.document(), value)
        self.assertEqual([c["path"] for c in captured.conversions()], [["finetune_params", "a", "x"], ["finetune_params", "b", "x"]])
        mixed = {"finetune_params": shared, "params": shared}
        error = self.assert_code("tuple_not_supported", source.capture_source, mixed)
        self.assertEqual(error.path, ("params", "x"))

    def test_shared_subtree_depth_rechecked_at_each_location(self):
        shared = [0]
        self.assertEqual(identity.encode_recipe([shared, [shared]], max_depth=3).value(), [[0], [[0]]])
        error = self.assert_code("depth_limit", identity.encode_recipe, [shared, [shared]], max_depth=2)
        self.assertEqual(error.path, (1, 0, 0))
        nested = shared
        for _ in range(127):
            nested = [nested]
        self.assert_code("depth_limit", source.capture_source, [shared, nested])

    def test_source_cap_boundary_includes_conversion_records(self):
        shared = {"x": (1, 2.0)}
        value = {"finetune_params": {"a": shared, "b": shared}, "label": "é"}
        captured = source.capture_source(value)
        size = len(identity.encode_recipe(json.loads(captured.payload)).canonical)
        self.assertEqual(source.capture_source(value, max_bytes=size).payload, captured.payload)
        self.assert_code("size_limit", source.capture_source, value, max_bytes=size - 1)

    def test_huge_integer_refuses_without_raw_conversion_error(self):
        self.assert_code("invalid_json", identity.encode_recipe, {"a": 10 ** 4300})
        self.assert_code("invalid_json", identity.decode_json, b"1" * 4301)

    def test_golden_literal_bytes_and_digest(self):
        encoded = identity.encode_recipe({"a": 1, "b": [True, None, 1.5, "x"]})
        self.assertEqual(encoded.profile, "nirs4all.realized-recipe.ordered-json.v1")
        self.assertEqual(encoded.payload, b'{"a":1,"b":[true,null,1.5,"x"]}')
        self.assertEqual(encoded.canonical, b'nirs4all.realized-recipe.ordered-json.v1\x00["mapping",[["a",["int","1"]],["b",["list",[["bool",true],["null"],["float","0x1.8000000000000p+0"],["string","x"]]]]]]')
        self.assertEqual(encoded.digest, "sha256:5dffc191f8bea6f8fa4e0e386293a936fa201d0a7273043567867a3f7f76344c")

    def test_canonical_expansion_cap_and_exact_boundary(self):
        value = [0] * 40
        encoded = identity.encode_recipe(value)
        self.assertLess(len(encoded.payload), 120)
        self.assertGreater(len(encoded.canonical), 120)
        self.assert_code("size_limit", identity.encode_recipe, value, max_bytes=120)
        size = len(encoded.canonical)
        self.assertEqual(identity.encode_recipe(value, max_bytes=size).canonical, encoded.canonical)
        self.assert_code("size_limit", identity.encode_recipe, value, max_bytes=size - 1)

    def test_depth_exact_boundary(self):
        value = 0
        for _ in range(128):
            value = [value]
        self.assertEqual(identity.encode_recipe(value).value(), value)
        self.assert_code("depth_limit", identity.encode_recipe, [value])

    def test_limit_arguments_cannot_enlarge_profile(self):
        for kwargs in ({"max_bytes": identity.MAX_RECIPE_BYTES + 1}, {"max_bytes": True}, {"max_bytes": 0}, {"max_depth": True}):
            self.assert_code("limit_invalid", identity.encode_recipe, {}, **kwargs)

    def test_invalid_key_reports_raw_container(self):
        for function in (identity.encode_recipe, source.capture_source):
            error = self.assert_code("invalid_unicode", function, {"a": {"\ud800": 1}})
            self.assertEqual(error.path, ("a",))

    def test_float_subclass_refused(self):
        class FloatSubclass(float):
            pass

        self.assert_code("value_not_supported", identity.encode_recipe, {"a": FloatSubclass(1.0)})

    def test_codec_key_unicode_refusal(self):
        self.assert_code("invalid_unicode", identity.encode_recipe, {"a": {"\ud800": 1}})

    def test_digest_grammar_exact(self):
        encoded = identity.encode_recipe({})
        for digest in (encoded.digest + "\n", encoded.digest.upper(), encoded.digest.encode(), None):
            self.assert_code("digest_invalid", identity.verify_recipe, encoded.payload, digest)

    def tamper(self, captured, **changes):
        value = json.loads(captured.payload)
        value.update(changes)
        payload = identity.encode_recipe(value).payload
        return payload, "sha256:" + hashlib.sha256(payload).hexdigest()

    def test_envelope_exact_fields(self):
        captured = source.capture_source({})
        self.assert_code("source_schema_invalid", source.read_source, *self.tamper(captured, extra="not allowed"))

    def test_constructor_envelope_requires_parameter_dict(self):
        captured = source.capture_source([])
        self.assert_code("source_schema_invalid", source.read_source, *self.tamper(captured, representation="constructor_parameters"))

    def test_conversion_order_rechecked(self):
        captured = source.capture_source({"finetune_params": {"a": (1, 2), "b": (3, 4)}})
        conversions = captured.conversions()
        self.assert_code("source_conversion_invalid", source.read_source, *self.tamper(captured, conversions=list(reversed(conversions))))

    def test_nonjson_original_text_must_be_unavailable(self):
        captured = source.capture_source({})
        self.assert_code("source_schema_invalid", source.read_source, *self.tamper(captured, original_text="{}"))

    def test_mixed_range_endpoint_types_are_float(self):
        for endpoints in ((1, 2.0), (1.0, 2), (-0.0, 1)):
            captured = source.capture_source({"finetune_params": {"x": endpoints}})
            self.assertEqual(captured.normalized_document()["finetune_params"]["x"]["type"], "float")
            self.assertEqual([type(v) for v in captured.document()["finetune_params"]["x"]], [type(v) for v in endpoints])

    def test_nonfinite_range_reports_raw_tuple(self):
        for endpoint in (float("inf"), float("-inf"), float("nan")):
            value = {"steps": [{"finetune_params": {"x": (1, endpoint)}}]}
            error = self.assert_code("nonfinite_number", source.capture_source, value)
            self.assertEqual(error.path, ("steps", 0, "finetune_params", "x"))


if __name__ == "__main__":
    unittest.main()
