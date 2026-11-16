"""Generate redistributable multimodal teaching files through the SDK synthesis host.

This is offline example authoring, not a browser or production numerical kernel.
No dataset-wide learned normalization, target-based filtering, or model fitting
is performed. Generated files use the same IO manifest as user imports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np

from nirs4all.synthesis.generator import SyntheticNIRSGenerator


def write_json(path: Path, value: object) -> None:
    """Write deterministic readable UTF-8 JSON."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Write exact ID strings and round-trippable float cells."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_png(path: Path, pixels: np.ndarray) -> None:
    """Encode a deterministic 8-bit grayscale PNG using only standard codecs."""
    height, width = pixels.shape

    def chunk(name: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + name + body + struct.pack(">I", zlib.crc32(name + body))

    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def spectral_source(folder: Path, name: str, label: str, values: np.ndarray, axis: np.ndarray, unit: str, kind: str, ids: list[str]) -> dict[str, object]:
    """Write a source in deliberately reversed row order to exercise ID joins."""
    features = [format(float(value), ".12g") for value in axis]
    write_csv(folder / f"{name}.csv", ["sample_id", *features], [[ids[row], *values[row].tolist()] for row in reversed(range(len(ids)))])
    return {"id": name, "label": label, "modality": "spectroscopy", "profile": "numeric_csv", "file": f"{name}.csv", "sampleId": "sample_id", "features": features,
            "axis": {"values": axis.tolist(), "unit": unit, "kind": kind}}


def generate_case(output: Path, case: str, seed: int) -> None:
    """Create one fixed-size teaching preset with recorded latent relationships."""
    names = {"spectroscopy": "spectroscopy-complementary", "genetics": "genetics-replicates", "images": "images-spatial"}
    folder = output / names[case]
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    n = 120
    ids = [f"sample-{index:03d}" for index in range(n)]
    wavelengths = np.linspace(1000.0, 2300.0, 64)
    generator = SyntheticNIRSGenerator(wavelengths=wavelengths, complexity="simple", random_state=seed)
    groups = [f"material-{index // 2:03d}" for index in range(n)] if case == "genetics" else None
    base_count = n // 2 if groups else n
    concentrations = rng.dirichlet(np.full(generator.library.n_components, 1.4), size=base_count)
    if groups:
        concentrations = np.repeat(concentrations, 2, axis=0)
    spectra, _ = generator.generate_from_concentrations(
        concentrations, include_instrument_effects=False, include_environmental_effects=False,
        include_scattering_effects=False, include_edge_artifacts=False,
    )
    shared = concentrations[:, 0]
    complementary = rng.uniform(-1.0, 1.0, n)
    target = 3.0 * shared + 0.8 * concentrations[:, 1]
    sources = [spectral_source(folder, "nirs", "NIRS", spectra, wavelengths, "nm", "wavelength", ids)]
    formulas: dict[str, object] = {"nirs": "SDK simple component library, shared Dirichlet concentrations; seeded Beer-Lambert/path-length/baseline/noise host",
                                   "base_target": "3*c0 + 0.8*c1; c are shared Dirichlet concentrations", "noise": "Independent Gaussian target noise sd=0.06"}
    if case == "spectroscopy":
        raman_axis = np.linspace(200.0, 1800.0, 48)
        mir_axis = np.linspace(4000.0, 600.0, 56)
        def peak(axis: np.ndarray, center: float, spread: float) -> np.ndarray:
            return np.exp(-0.5 * ((axis - center) / spread) ** 2)

        raman = (1.2 + complementary[:, None]) * peak(raman_axis, 820.0, 45.0) + shared[:, None] * peak(raman_axis, 1340.0, 75.0)
        raman += 0.03 * rng.normal(size=raman.shape) + rng.uniform(0.0, 0.4, n)[:, None]
        mir = 8.0 * concentrations[:, 1, None] * peak(mir_axis, 1650.0, 120.0) + 5.0 * shared[:, None] * peak(mir_axis, 3350.0, 180.0)
        mir += 0.08 * rng.normal(size=mir.shape) + rng.uniform(0.0, 0.8, n)[:, None]
        sources = [spectral_source(folder, "raman", "Raman", raman, raman_axis, "cm-1", "raman_shift", ids), sources[0],
                   spectral_source(folder, "mir", "MIR", mir, mir_axis, "cm-1", "wavenumber", ids)]
        target += 0.9 * complementary
        formulas.update({"raman": "(1.2+u)*Gaussian(820,45) + c0*Gaussian(1340,75) + baseline U(0,.4) + noise sd=.03",
                         "mir": "8*c1*Gaussian(1650,120) + 5*c0*Gaussian(3350,180) + baseline U(0,.8) + noise sd=.08",
                         "target_addition": "0.9*u, u~U(-1,1); no real-world ranking claim"})
    elif case == "genetics":
        markers_by_group = rng.binomial(2, 0.35, size=(n // 2, 16))
        markers = np.repeat(markers_by_group, 2, axis=0)
        target += 0.35 * markers[:, 0] - 0.25 * markers[:, 3] + 0.15 * markers[:, 7]
        # Both real VCF and matching dosage CSV are provided. The teaching
        # manifest uses VCF; switching to numeric_csv exercises the second route.
        feature_names = [f"marker-{index:02d}" for index in range(16)]
        write_csv(folder / "markers.csv", ["sample_id", *feature_names], [[ids[row], *markers[row].tolist()] for row in reversed(range(n))])
        with (folder / "markers.vcf").open("w", encoding="utf-8", newline="") as stream:
            stream.write("##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Synthetic diploid genotype\">\n")
            stream.write("\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *reversed(ids)]) + "\n")
            for column in range(16):
                calls = []
                for row in reversed(range(n)):
                    genotype = ["0/0", "0/1", "1/1"][int(markers[row, column])]
                    calls.append(genotype.replace("/", "|") if column % 2 else genotype)
                stream.write("\t".join(["1", str(1000 + 100 * column), feature_names[column], "A", "G", ".", "PASS", ".", "GT", *calls]) + "\n")
        sources.append({"id": "genetics", "label": "Genetic markers", "modality": "genetics", "profile": "vcf_gt_dosage", "file": "markers.vcf"})
        formulas.update({"markers": "16 independent Binomial(2,0.35) group-level dosages; identical genotypes in each replicate pair",
                         "target_addition": "0.35*g0 - 0.25*g3 + 0.15*g7; measurement replicates share concentration/markers and differ in spectral noise"})
    else:
        yy, xx = np.mgrid[-1.0:1.0:12j, -1.0:1.0:12j]
        spot = np.exp(-((xx - 0.25) ** 2 + (yy + 0.2) ** 2) / 0.12)
        stripes = 0.5 + 0.5 * np.sin(8.0 * xx)
        image_factor = rng.uniform(0.0, 1.0, n)
        target += 1.1 * image_factor
        mapping = []
        for index, sample_id in enumerate(ids):
            pixels = 24.0 + 150.0 * image_factor[index] * spot + 45.0 * shared[index] * stripes + rng.normal(0.0, 4.0, size=spot.shape)
            encoded = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
            name = f"{sample_id}.png"
            write_png(folder / name, encoded)
            mapping.append([sample_id, name])
        write_csv(folder / "images.csv", ["sample_id", "file"], list(reversed(mapping)))
        sources.append({"id": "image", "label": "Grayscale images", "modality": "image", "profile": "png_manifest", "file": "images.csv", "shape": [12, 12, 1]})
        formulas.update({"image": "round(clip(24 + 150*v*GaussianSpot + 45*c0*Stripes + pixel noise sd=4,0,255)); fixed physical encoding range, no learned global normalization",
                         "target_addition": "1.1*v, v~U(0,1); geometry is HWC 12x12x1"})
    target += rng.normal(0.0, 0.06, size=n)
    metadata_rows = [[sample_id, float(target[index]), *( [groups[index]] if groups else [] ),
                      "test" if (index // 2 if groups else index) % 5 == 0 else "train"] for index, sample_id in enumerate(ids)]
    metadata_header = ["sample_id", "target", *(["group_id"] if groups else []), "partition"]
    write_csv(folder / "metadata.csv", metadata_header, metadata_rows)
    provenance = {"license": "CC0-1.0", "seed": str(seed), "n_samples": n, "generator_owner": "nirs4all Python SDK synthesis host",
                  "generator_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "component_names": generator.library.component_names, "formulas": formulas,
                  "partition": "Every fifth declared material group/sample is retained test; fixed independently of X/y; group pair never split",
                  "training_policy": "No model fitted during generation. Hyperparameters/epochs fixed by the teaching recipe; results are not guaranteed."}
    manifest = {"schema": "nirs4all.multimodal-import.v1", "caseId": case, "name": names[case],
                "metadata": {"file": "metadata.csv", "sampleId": "sample_id", "target": "target", "targetUnit": "a.u.",
                             **({"group": "group_id"} if groups else {}), "partition": "partition"},
                "sources": sources, "provenance": provenance}
    write_json(folder / "manifest.json", manifest)
    (folder / "README.txt").write_text("CC0-1.0 synthetic teaching data. Unzip, then select manifest.json and every CSV/VCF/PNG file in the laboratory.\n"
        "Replace declared file names/columns/axes in manifest.json for your own data. Keep exact unique sample IDs.\n"
        "The metadata establishes canonical identity order. Source files deliberately reverse their rows.\n"
        "Missing genotype calls are preserved but training is refused by the complete-data profile. No imputation or resize occurs.\n", encoding="utf-8")
    inventory = sorted(path.name for path in folder.iterdir() if path.suffix in {".json", ".csv", ".vcf", ".png"})
    write_json(folder / "files.json", {"schema": "nirs4all.teaching-files.v1", "files": inventory})
    with zipfile.ZipFile(output / f"{names[case]}.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            info = zipfile.ZipInfo(path.name, (2026, 11, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


def main() -> None:
    """Write the three fixed presets under an explicit destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = Path(__file__).resolve().parents[2] / "nirs4all" / "synthesis" / "generator.py"
    import inspect
    actual = Path(inspect.getfile(SyntheticNIRSGenerator)).resolve()
    if actual != expected:
        raise RuntimeError(f"Expected isolated SDK source {expected}, imported {actual}")
    for offset, case in enumerate(("spectroscopy", "genetics", "images")):
        generate_case(args.output, case, 20261116 + offset)
    print(f"Generated three CC0 teaching presets through SDK source {actual}")


if __name__ == "__main__":
    main()
