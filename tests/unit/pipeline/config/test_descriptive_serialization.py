"""Serialization-only regressions; no SDK initialization or scientific runtime.

Run this file directly with Python -I -S -B. Each unittest launches a fresh,
guarded child loading only component_serialization.py and synthetic dependencies.
The synthetic numpy/sklearn/n4m modules below provide structure, never numerics.
This harness does not qualify those frameworks, PipelineConfigs, or run().
"""

import copy
import hashlib
import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
import threading
import traceback
import types
import unittest
from enum import Enum
from functools import partial
from pathlib import Path

SOURCE = Path(os.environ.get("PHASE7_SOURCE", Path(__file__).resolve().parents[4] / "nirs4all/pipeline/config/component_serialization.py"))
REF = "phase7_marker.callback"
ALIAS = "phase7_alias.exported"
SAFE = "phase7_safe.Holder"
DESCRIPTIVE = ("name", "description", "metadata", "_metadata_", "_tags_")
EVENTS = {"imports": [], "constructors": 0, "calls": 0}
CHILD_THREAD_ENV = dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")


class GuardViolation(BaseException):
    """Fail closed, outside product Exception/ImportError recovery."""


class SerializationGuard(importlib.abc.MetaPathFinder):
    """Deny real scientific imports, runtime entry and unguarded subprocesses.

    This is deliberately narrower than a general SDK test environment. The
    import allowlist applies process-wide, including importlib and new threads;
    tracing is an additional Python-call tripwire, not the native-code barrier.
    Site startup is disabled, and native scientific packages are never loaded.
    """

    def __init__(self):
        self.allowed = set(sys.stdlib_module_names) | {"phase7_source", "phase7_safe", "phase7_absent"}
        self.synthetic = {"numpy", "sklearn", "n4m", "flax", "phase7_marker", "phase7_alias",
                          "phase7_runtime_error", "phase7_system_exit", "phase7_absent", "v1"}
        self.denied_calls = {"fit", "partial_fit", "fit_transform", "fit_predict", "transform", "predict",
                             "predict_proba", "train", "train_step", "benchmark"}
        self.stdlib = Path(sysconfig.get_path("stdlib")).resolve()
        self.script = str(Path(__file__).resolve())
        self.worker = "--worker" in sys.argv
        self.proof = {}

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root not in self.allowed | self.synthetic or root in {"ctypes", "_ctypes"}:
            raise GuardViolation(f"blocked import: {fullname}")
        if root in self.synthetic:
            raise GuardViolation(f"unregistered synthetic module: {fullname}")
        return None

    def trace(self, frame, event, arg):
        if event == "call" and frame.f_code.co_name in self.denied_calls:
            raise GuardViolation(f"blocked runtime entry: {frame.f_code.co_name}")
        return self.trace

    def audit(self, event, args):
        if event.startswith(("socket.", "ctypes.", "os.exec", "os.spawn")) or event in {"os.system", "os.fork", "os.posix_spawn"}:
            raise GuardViolation(f"blocked process/native/network event: {event}")
        if event == "subprocess.Popen":
            command = args[1]
            expected = [sys.executable, "-I", "-S", "-B", self.script, "--worker"]
            if self.worker or not isinstance(command, list) or command[:6] != expected or len(command) != 7:
                raise GuardViolation("child must restart this guarded harness with -I -S -B")

    def install(self):
        assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert os.environ.get(key) == "1", key
        assert not any(name.partition(".")[0] in self.synthetic for name in sys.modules)
        sys.meta_path.insert(0, self)
        sys.addaudithook(self.audit)
        sys.settrace(self.trace)
        threading.settrace(self.trace)
        self.verify()

    def verify(self):
        def check():
            checks = []
            for module in ("numpy", "sklearn", "nirs4all", "ctypes"):
                try:
                    importlib.import_module(module)
                except GuardViolation:
                    checks.append(module)
                else:
                    raise AssertionError(f"guard allowed {module}")

            def fit():
                raise AssertionError("fit body must never execute")

            try:
                fit()
            except GuardViolation:
                checks.append("fit_before_body")
            return checks

        self.proof["main"] = check()
        outcomes = []
        failures = []

        def in_thread():
            try:
                outcomes.append(check())
            except BaseException as exc:
                failures.append(repr(exc))

        thread = threading.Thread(target=in_thread)
        thread.start()
        thread.join()
        assert not failures and outcomes == [self.proof["main"]], failures
        self.proof["worker_thread"] = outcomes[0]
        # A trace exception disables tracing in that thread; restore it after
        # the tripwire self-test. The process-wide import/audit guards persist.
        sys.settrace(self.trace)


def synthetic_module(name, **members):
    module = types.ModuleType(name)
    module.__phase7_synthetic__ = True
    module.__dict__.update(members)
    module.__path__ = []
    sys.modules[name] = module
    return module


def fixtures():
    """Register structural doubles before the source can request dependencies."""
    class Holder:
        def __init__(self, name=None, cfg=None, items=None, func=None):
            self.name, self.cfg, self.items, self.func = name, cfg, items, func

    Holder.__module__, Holder.__qualname__ = "phase7_safe", "Holder"

    class StringHolder:
        def __init__(self, name: str | None = None):
            self.name = name

    StringHolder.__module__, StringHolder.__qualname__ = "phase7_safe", "StringHolder"

    class TupleHolder:
        def __init__(self, items=()):
            self.items = items

    TupleHolder.__module__, TupleHolder.__qualname__ = "phase7_safe", "TupleHolder"

    def factory(*args, **kwargs):
        EVENTS["calls"] += 1
        raise AssertionError("factory must remain deferred")

    factory.__module__, factory.__name__ = "phase7_safe", "factory"
    factory.framework = "synthetic"
    safe = synthetic_module("phase7_safe", Holder=Holder, StringHolder=StringHolder, TupleHolder=TupleHolder, factory=factory)

    class Flag(Enum):
        ONE = 1

    Flag.__module__, Flag.__qualname__ = "phase7_safe", "Flag"
    safe.Flag = Flag

    class Array:
        def __init__(self, values, dtype="int64"):
            self.values = copy.deepcopy(values)
            self.dtype = types.SimpleNamespace(kind="i")
            self.shape = (len(values),)

        def tolist(self):
            return copy.deepcopy(self.values)

    synthetic_module("numpy", ndarray=Array, array=Array)
    synthetic_module("sklearn")

    class Ensemble:
        def __init__(self, estimators=None, final_estimator=None, cfg=None):
            self.estimators, self.final_estimator, self.cfg = estimators, final_estimator, cfg
            self.nested = {}

        def set_params(self, **kwargs):
            self.nested.update(kwargs)
            return self

    Ensemble.__module__, Ensemble.__qualname__ = "sklearn.ensemble", "StackingRegressor"
    synthetic_module("sklearn.ensemble", StackingRegressor=Ensemble, StackingClassifier=type("SC", (), {}),
                     VotingRegressor=type("VR", (), {}), VotingClassifier=type("VC", (), {}))
    synthetic_module("flax")
    synthetic_module("flax.linen", Module=type("Module", (), {}))
    synthetic_module("n4m")

    class NativeMethod:
        pass

    class Role(NativeMethod):
        _method_id = "synthetic"

        def __init__(self, name=None):
            self.name = name

    Role.__module__, Role.__qualname__ = "n4m.roles", "Role"

    def method_class(identifier):
        assert identifier == "synthetic"
        return Role

    synthetic_module("n4m.roles", NativeMethod=NativeMethod, method_class=method_class, Role=Role)
    synthetic_module("n4m.transform")
    synthetic_module("n4m.transform.scatter", SNV=Holder)
    synthetic_module("phase7_source")
    synthetic_module("phase7_source.portable_method_specs", PORTABLE_METHOD_SPECS={})
    return safe


class MarkerLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Real import dispatch with synthetic modules and separate effect counters."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"phase7_marker", "phase7_alias", "phase7_runtime_error", "phase7_system_exit", "phase7_absent", "v1"}:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ in {"phase7_absent", "v1"}:
            raise ModuleNotFoundError(f"synthetic missing module: {module.__name__}")
        EVENTS["imports"].append(module.__name__)
        module.__phase7_synthetic__ = True
        if module.__name__ == "phase7_runtime_error":
            raise RuntimeError("synthetic import-time error")
        if module.__name__ == "phase7_system_exit":
            raise SystemExit("synthetic import-time exit")
        if module.__name__ == "phase7_alias":
            module.exported = importlib.import_module("phase7_marker").callback
            return

        def callback(*args, **kwargs):
            EVENTS["calls"] += 1
            raise AssertionError("marker callback must never be invoked")

        class Zero:
            def __init__(self):
                EVENTS["constructors"] += 1

        class Required:
            def __init__(self, required):
                EVENTS["constructors"] += 1

        for name, obj in (("callback", callback), ("Zero", Zero), ("Required", Required)):
            obj.__module__, obj.__qualname__ = module.__name__, name
            if name == "callback":
                obj.__name__ = name
            setattr(module, name, obj)


def load_source():
    fixtures()
    sys.meta_path.insert(0, MarkerLoader())
    name = "phase7_source.component_serialization"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_no_effects():
    assert EVENTS == {"imports": [], "constructors": 0, "calls": 0}, EVENTS


def callbacks(value):
    if isinstance(value, dict):
        return sum(callbacks(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(callbacks(v) for v in value)
    return int(callable(value) and getattr(value, "__module__", None) == "phase7_marker")


def execute_case(case, module):
    ser, des = module.serialize_component, module.deserialize_component
    safe = sys.modules["phase7_safe"]
    if case.startswith("literal:"):
        _, operation, shape, key, reference = case.split(":", 4)
        ref = {"callback": REF, "constructor": "phase7_marker.Zero", "missing": "phase7_marker.missing",
               "required": "phase7_marker.Required", "runtime": "phase7_runtime_error.value",
               "exit": "phase7_system_exit.value", "relative": ".model.v2", "dots": "...",
               "role": "n4m:unknown", "plain": "My step", "version": "v1.2"}[reference]
        leaf = {key: {"nested": [ref]} if key in DESCRIPTIVE[2:] else ref}
        shapes = {
            "bare": leaf, "class": {"class": SAFE, **leaf}, "step": {"step": {"class": SAFE}, **leaf},
            "transform": {"transform": {"class": SAFE}, **leaf}, "custom": {"custom": {"class": SAFE}, **leaf},
            "model": {"model": {"class": SAFE}, **leaf},
            "nested": {"model": {"model": {"class": SAFE}, **leaf}},
            "branch": {"branch": {"z_first": [{"model": {"class": SAFE}, **leaf}], "a_second": []}},
            "choice": {"_or_": [{"class": SAFE}], **leaf},
            "concat": {"concat_transform": {"transformers": [{"class": SAFE}], **leaf}},
            "pipeline": {"pipeline": [{"class": SAFE}], **leaf}, "steps": {"steps": [{"class": SAFE}], **leaf},
        }
        value = shapes[shape]
        before = copy.deepcopy(value)
        result = (ser if operation == "ser" else des)(value)
        assert_no_effects()
        assert value == before
        if operation == "ser" or shape == "bare":
            assert result == value
        assert result is not value
        return

    if case == "literal_containers":
        value = {"metadata": {"class": "phase7_marker.Zero", "enum": "phase7_marker.Zero", "value": REF,
                              "params": {"name": REF}, "nested": [{"function": REF}], True: REF}}
        before = copy.deepcopy(value)
        for operation in (ser, des):
            result = operation(value)
            assert result == before
            result["metadata"]["nested"][0]["function"] = "changed"
            assert value == before
        assert_no_effects()
    elif case == "literal_normalization":
        array = sys.modules["numpy"].ndarray([1, 2])
        value = {"name": safe.Holder, "metadata": {"tuple": (REF, None, 7), "array": array, "flag": safe.Flag.ONE}}
        result = ser(value)
        assert result["name"] == SAFE
        assert result["metadata"]["tuple"] == [REF, None, 7]
        assert result["metadata"]["array"]["class"] == "numpy.array"
        assert result["metadata"]["array"]["params"]["object"] == [1, 2]
        assert result["metadata"]["flag"] == {"enum": "phase7_safe.Flag", "value": 1}
        json.dumps(result)
        assert_no_effects()
    elif case.startswith("argument:"):
        _, mode, shape = case.split(":")
        leaf = {"name": ALIAS}
        shapes = {"name": {"params": leaf}, "cfg": {"params": {"cfg": leaf}},
                  "items": {"params": {"items": [leaf]}}, "args": {"args": [leaf]},
                  "grid": {"_grid_": {"cfg": [leaf]}}, "zip": {"_zip_": {"cfg": [leaf]}},
                  "or": {"params": {"cfg": {"_or_": [leaf]}}},
                  "finetune": {"finetune_params": {"model_params": leaf}},
                  "train": {"train_params": leaf}}
        value = shapes[shape]
        result = (ser if mode == "ser" else des)(value)
        if mode == "ser":
            assert REF in json.dumps(result) and ALIAS not in json.dumps(result), result
        else:
            assert callbacks(result) == 1, result
        assert EVENTS["imports"] == ["phase7_alias", "phase7_marker"]
        assert EVENTS["constructors"] == EVENTS["calls"] == 0
    elif case == "parameter_or":
        value = {"_or_": [ALIAS], "param": "name", "model": {"class": SAFE}}
        assert ser(value) == {"_or_": [REF], "param": "name", "model": {"class": SAFE}}
        restored = des(value)  # This harness does not expand generators.
        assert restored["_or_"][0] is sys.modules["phase7_marker"].callback
        assert restored["param"] == "name" and isinstance(restored["model"], safe.Holder)
        assert EVENTS["constructors"] == EVENTS["calls"] == 0
    elif case == "parameter_or_hash_edge":
        value = {"_or_": {"cfg": [{"name": ALIAS}]}}
        assert ser(value) == value  # No Argument key on this path.
        assert des({"params": value})["params"]["_or_"]["cfg"][0]["name"] is sys.modules["phase7_marker"].callback
    elif case.startswith("explicit:"):
        key = case.partition(":")[2]
        if key == "bare":
            value = "phase7_marker.Zero"
        elif key == "function":
            value = {key: REF}
        else:
            value = {key: "phase7_marker.Zero"}
        result = des(value)
        assert EVENTS["imports"] == ["phase7_marker"]
        assert EVENTS["constructors"] == (0 if key == "function" else 1)
        if key == "function":
            assert result is sys.modules["phase7_marker"].callback
        assert EVENTS["calls"] == 0
    elif case.startswith("constructor:"):
        shape = case.partition(":")[2]
        params = {"name": REF} if shape == "name" else {"cfg": {"name": REF}} if shape == "cfg" else {"items": [{"name": REF}]}
        result = des({"class": SAFE, "params": params})
        assert callbacks(vars(result)) == 1
        assert EVENTS["constructors"] == EVENTS["calls"] == 0
    elif case == "argument_component_metadata":
        value = {"params": {"cfg": {"class": SAFE, "name": REF, "metadata": {"function": REF}}}}
        assert ser(value) == value
        assert isinstance(des(value)["params"]["cfg"], safe.Holder)
        assert_no_effects()
    elif case.startswith("argument_envelope:"):
        key = case.partition(":")[2]
        declaration = REF if key == "function" else SAFE
        value = {"params": {"cfg": {key: declaration, "name": REF, "description": ALIAS, "metadata": {"nested": [REF]}}}}
        before = copy.deepcopy(value)
        serialized = ser(value)
        assert serialized == value
        restored = des(serialized)["params"]["cfg"]
        if key == "function":
            assert restored is sys.modules["phase7_marker"].callback
            assert EVENTS == {"imports": ["phase7_marker"], "constructors": 0, "calls": 0}
        else:
            assert isinstance(restored, safe.Holder)
            assert_no_effects()
        assert value == before
    elif case == "live_instance":
        value = safe.Holder(name=ALIAS, cfg={"name": ALIAS}, items=[{"name": ALIAS}])
        result = des(ser(value))
        assert callbacks(vars(result)) == 3
        assert EVENTS["calls"] == EVENTS["constructors"] == 0
    elif case == "literal_live_instance":
        result = ser({"metadata": safe.Holder(cfg={"name": ALIAS})})
        assert result["metadata"]["params"]["cfg"]["name"] == REF
        assert EVENTS["imports"] == ["phase7_alias", "phase7_marker"]
    elif case == "partial":
        callback = importlib.import_module("phase7_marker").callback
        value = partial(callback, {"name": ALIAS}, cfg={"name": ALIAS})
        serialized = ser(value)
        assert serialized["args"][0]["name"] == REF
        assert serialized["params"]["cfg"]["name"] == REF
        restored = des(serialized)
        assert restored.func is callback and restored.args[0]["name"] is callback and restored.keywords["cfg"]["name"] is callback
        assert EVENTS["calls"] == EVENTS["constructors"] == 0
    elif case == "function_args":
        result = des({"function": REF, "args": [{"name": REF}], "params": {"cfg": {"name": REF}}})
        assert isinstance(result, partial) and callbacks(result.args) == callbacks(result.keywords) == 1
        assert EVENTS["calls"] == EVENTS["constructors"] == 0
    elif case == "deferred_factory":
        result = des({"function": "phase7_safe.factory", "params": {"cfg": {"name": REF}}})
        assert result["func"] is safe.factory and result["framework"] == "synthetic"
        assert callbacks(result["params"]) == 1 and EVENTS["calls"] == 0
    elif case == "meta_estimator":
        ensemble = sys.modules["sklearn.ensemble"].StackingRegressor
        value = ensemble(estimators=[("plain.name", safe.Holder(cfg={"name": ALIAS}))],
                         final_estimator=safe.Holder(name=ALIAS), cfg={"name": ALIAS})
        serialized = ser(value)
        assert serialized["params"]["estimators"][0][0] == "plain.name"
        assert serialized["params"]["cfg"]["name"] == REF
        restored = des(serialized)
        assert callbacks(vars(restored.estimators[0][1])) == 1
        assert callbacks(vars(restored.final_estimator)) == callbacks(restored.cfg) == 1
        extra = des({"class": "sklearn.ensemble.StackingRegressor", "params": {"final_estimator__cfg": {"name": REF}}})
        assert callbacks(extra.nested) == 1
        assert EVENTS["calls"] == EVENTS["constructors"] == 0
    elif case == "role":
        role = sys.modules["n4m.roles"].Role(name=ALIAS)
        serialized = ser(role)
        assert serialized == {"class": "n4m:synthetic", "params": {"name": REF}}
        assert des(serialized).name is sys.modules["phase7_marker"].callback
        assert isinstance(des("n4m:synthetic"), type(role))
    elif case == "alias":
        assert isinstance(des({"class": "n4m.SNV"}), safe.Holder)
        assert isinstance(des("n4m.SNV"), safe.Holder)
        assert_no_effects()
    elif case == "infer_type":
        assert des(REF, infer_type=str, strict_imports=True) == REF
        assert des([REF], infer_type=tuple)[0] is sys.modules["phase7_marker"].callback
        assert des({"class": "phase7_safe.StringHolder", "params": {"name": REF}}).name == REF
    elif case == "tuple_argument":
        value = {"class": "phase7_safe.TupleHolder", "params": {"items": [{"name": REF}]}}
        before = copy.deepcopy(value)
        restored = des(value)
        assert isinstance(restored.items, tuple) and len(restored.items) == 1
        assert callbacks(restored.items) == 1, "tuple argument lost callable resolution"
        assert restored.items[0]["name"] is sys.modules["phase7_marker"].callback
        assert value == before
        assert EVENTS == {"imports": ["phase7_marker"], "constructors": 0, "calls": 0}
    elif case == "strict":
        assert des("phase7_absent.value") == "phase7_absent.value"
        try:
            des("phase7_absent.value", strict_imports=True)
        except ValueError:
            pass
        else:
            raise AssertionError("strict import must refuse missing executable")
        assert_no_effects()
    elif case == "finetune_ranges":
        result = ser({"finetune_params": {"model_params": {"count": (1, 4), "category": [1, 4]}}})
        assert result["finetune_params"]["model_params"] == {"count": {"type": "int", "min": 1, "max": 4}, "category": [1, 4]}
        assert_no_effects()
    elif case == "identity":
        left, right = [{"custom": {"class": SAFE}, "name": ALIAS}], [{"custom": {"class": SAFE}, "name": REF}]
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        assert digest(ser(left)) != digest(ser(right))
        assert ser(left) == [ser(step) for step in left]
        assert ser({"name": "Plant.v2"}) == {"name": "Plant.v2"}
        assert_no_effects()
    else:
        raise AssertionError(case)
    assert EVENTS["calls"] == 0


CASES = [f"literal:{op}:{shape}:{key}:callback" for op in ("ser", "des")
         for shape in ("bare", "class", "step", "transform", "custom", "model", "nested", "branch", "choice", "concat", "pipeline", "steps")
         for key in DESCRIPTIVE]
CASES += [f"literal:{op}:model:name:{ref}" for op in ("ser", "des")
          for ref in ("constructor", "missing", "required", "runtime", "exit", "relative", "dots", "role", "plain", "version")]
CASES += [f"argument:{op}:{shape}" for op in ("ser", "des")
          for shape in ("name", "cfg", "items", "args", "grid", "zip", "or", "finetune", "train")]
CASES += [f"explicit:{key}" for key in ("class", "function", "instance", "bare")]
CASES += [f"constructor:{shape}" for shape in ("name", "cfg", "items")]
CASES += ["literal_containers", "literal_normalization", "parameter_or", "parameter_or_hash_edge", "argument_component_metadata",
          "live_instance", "literal_live_instance", "partial", "function_args", "deferred_factory", "meta_estimator",
          "role", "alias", "infer_type", "strict", "finetune_ranges", "identity"]
CASES += ["tuple_argument", "argument_envelope:function", "argument_envelope:instance"]


class TestDescriptiveSerialization(unittest.TestCase):
    """One fresh guarded process per semantic counterexample/control."""


def make_test(case):
    def test(self):
        command = [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), "--worker", case]
        child_env = os.environ.copy()
        child_env.update(CHILD_THREAD_ENV)
        result = subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=15, check=False)
        print(result.stdout, end="", flush=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout.splitlines()[-1])
        self.assertTrue(record["guard"]["main"] == record["guard"]["worker_thread"])
        self.assertEqual(record["thread_environment"], CHILD_THREAD_ENV)
    return test


for index, case in enumerate(CASES):
    setattr(TestDescriptiveSerialization, f"test_{index:03d}_{case.replace(':', '_')}", make_test(case))


if __name__ == "__main__":
    guard = SerializationGuard()
    guard.install()
    if "--worker" in sys.argv:
        assert len(sys.argv) == 3
        record = {"case": sys.argv[2], "pid": os.getpid(), "ppid": os.getppid(), "guard": guard.proof,
                  "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  "thread_environment": {key: os.environ.get(key) for key in CHILD_THREAD_ENV}}
        code = 0
        try:
            execute_case(sys.argv[2], load_source())
        except BaseException:
            code = 1
            traceback.print_exc()
        record.update(events=EVENTS, exit_code=code)
        print(json.dumps(record), flush=True)
        raise SystemExit(code)
    print(json.dumps({"parent_pid": os.getpid(), "guard": guard.proof, "cases": len(CASES),
                      "python": sys.version, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}), flush=True)
    unittest.main(verbosity=2)
