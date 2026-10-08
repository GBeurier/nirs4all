"""Check the built SDK's installed origins, dependencies and native ABI."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from importlib.metadata import version
from pathlib import Path

from packaging.version import Version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    assignments = ast.parse((source / "nirs4all/__init__.py").read_text(encoding="utf-8"))
    expected = next(
        ast.literal_eval(node.value)
        for node in assignments.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    )
    distributions = {name: version(name) for name in ("nirs4all", "dag-ml", "dag-ml-data", "nirs4all-core", "nirs4all-methods", "nirs4all-io")}
    assert distributions["nirs4all"] == expected, distributions
    assert Version("0.3.41") <= Version(distributions["dag-ml"]) < Version("0.4"), distributions
    assert Version("0.4.5") <= Version(distributions["nirs4all-core"]) < Version("0.5"), distributions
    origins = {}
    for name in ("nirs4all", "dag_ml", "nirs4all_core", "n4m"):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        assert not origin.is_relative_to(source), f"Imported source checkout instead of installed package: {origin}"
        origins[name] = str(origin)
    sdk = importlib.import_module("nirs4all")
    assert sdk.__version__ == expected
    assert callable(sdk.run)
    assert importlib.import_module("dag_ml").version() == distributions["dag-ml"]
    methods = importlib.import_module("n4m")
    assert methods.abi_version()[:2] == (2, 17)
    assert methods.version().split("+abi.")[0] == distributions["nirs4all-methods"]
    library = Path(methods.library_path()).resolve()
    assert library.is_file() and not library.is_relative_to(source)
    with library.open("rb") as stream:
        library_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps({"distributions": distributions, "origins": origins, "methods_abi": methods.abi_version(), "methods_library": str(library), "methods_sha256": library_sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
