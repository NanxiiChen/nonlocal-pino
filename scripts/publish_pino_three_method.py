"""Copy reviewed, audited comparison figures into the local README/site collection.

This does not render, train, deploy or modify defaults. A selected temporal case
can be restyled only with its explicit previous provenance and unchanged errors.
An explicitly selected Electropolish15k study may supersede its10k study.
Inspect the figures before supplying --visual-review. Full provenance stays at
its direct external path; only the compact record and four final assets enter
the repository. Repeating the same reviewed publication is read-only.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import struct


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "channel_nsch2d": (20000, 500, 1, 3),
    "electropolish2d": (10000, 250, 64, 64),
    "icemelting3d": (10000, 250, 64, 64),
    "confined_ch2d": (50000, 500, 8, 8),
}


def validate_budget_extension(data, previous):
    """Keep budget extension separate from the unchanged-data restyling gate."""
    if (data['comparison_kind'] != 'time_fusion' or data['example'] != 'electropolish2d'
            or previous['comparison_kind'] != 'time_fusion' or previous['example'] != data['example']
            or previous['budget'] != 10000 or data['budget'] != 15000
            or data['steps'] != list(range(0, 15001, 250))):
        raise ValueError('Only the explicit Electropolish10k-to15k temporal extension is supported')
    for key in ('monitor_ids', 'test_ids', 'fields', 'seed'):
        if data[key] != previous[key]:
            raise ValueError('Budget extension must retain the comparison scope: ' + key)
    # Older reports store the64individual paths, newer reports their common root.
    def references(value):
        if isinstance(value, list):
            return value
        return [str(Path(value)/f'condition_{i:03d}/local101.npz') for i in data['test_ids']]
    if references(data['reference']) != references(previous['reference']):
        raise ValueError('Budget extension changed numerical reference inputs')
    methods = ['decoder_film', 'joint_query']
    if [a['method'] for a in data['arms']] != methods or [a['method'] for a in previous['arms']] != methods:
        raise ValueError('Budget extension must retain both temporal methods')


def publish(source, visual_review, *, replace_provenance=None, supersede_provenance=None):
    if replace_provenance is not None and supersede_provenance is not None:
        raise ValueError('Restyling and budget extension are mutually exclusive')
    source = Path(source).resolve(strict=True)
    report = json.loads(source.read_text())
    data = report["data"]
    collection = "three_method"
    if report["schema"] == "pino_three_method_figure_v1":
        suffix, methods, cases, collection_limit = (
            "three_method", ["nonlocal", "local_ad", "fem"], CASES, 4 * 1024**2)
    elif report["schema"] == "pino_nonlocal_fem_figure_v1":
        from scripts.pino_comparison_data import SPINODAL_OMISSION
        suffix, methods, cases, collection_limit = (
            "nonlocal_fem", ["nonlocal", "fem"],
            {"confined_ch2d": CASES["confined_ch2d"]}, 4 * 1024**2)
        if (data["comparison_kind"] != "spinodal_nonlocal_fem"
                or data["omitted_method"] != SPINODAL_OMISSION):
            raise ValueError("Require the explicit user-authorized Spinodal AD omission")
    elif report["schema"] == "pino_time_fusion_figure_v1":
        collection = "time_fusion"
        suffix, methods, cases, collection_limit = (
            "time_fusion", ["decoder_film", "joint_query"],
            {**{name: CASES[name] for name in ("electropolish2d", "icemelting3d")},
             "particle_corrosion2d": (20000, 500, 64, 64)}, 2 * 1024**2)
        if (data["comparison_kind"] != "time_fusion"
                or data["monitor_ids"] != data["test_ids"]
                or data["pairing"]["shared_initial_backbone_equal"] is not True
                or data["pairing"]["all_actual_sampling_arrays_equal"] is not True
                or data["pairing"]["source_bytes_verified"] is not True):
            raise ValueError("Require paired full-test temporal controls")
    else:
        raise ValueError("Unsupported comparison-figure schema")
    if report["status"] != "passed" or data["status"] != "passed":
        raise ValueError("Require a completed, independently audited comparison figure")
    if data["example"] not in cases or not visual_review.strip():
        raise ValueError("Require a supported example and an actual visual-review note")
    budget, interval, monitors, tests = cases[data["example"]]
    extended_electro = (collection == 'time_fusion' and data['example'] == 'electropolish2d'
                       and data['budget'] == 15000)
    if extended_electro:
        budget = 15000
    if supersede_provenance is not None:
        if collection != 'time_fusion' or data['example'] != 'electropolish2d':
            raise ValueError('Budget extension is limited to the Electropolish temporal study')
        budget = 15000
    if (data["budget"] != budget or data["steps"] != list(range(0, budget + 1, interval))
            or len(data["monitor_ids"]) != monitors or len(data["test_ids"]) != tests
            or len(set(data["monitor_ids"])) != monitors or len(set(data["test_ids"])) != tests
            or [arm["method"] for arm in data["arms"]] != methods
            or set(report["figures"]) != {"convergence", "distribution"}):
        raise ValueError("Incomplete or mismatched methods, budget or cohort")
    manifest_path = ROOT / f"artifacts/pino_{collection}_figures.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["schema"] != f"pino_{collection}_publication_v1"
            or manifest["maximum_file_bytes"] != 1024**2
            or manifest["maximum_collection_bytes"] != collection_limit):
        raise ValueError("Unexpected publication schema or approved asset budget")
    names = [case["example"] for case in manifest["cases"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate selected examples")
    if extended_electro and supersede_provenance is None and replace_provenance is None:
        selected = [case for case in manifest['cases'] if case['example'] == data['example']]
        if len(selected) != 1 or selected[0]['source_provenance'] != str(source):
            raise ValueError('A new15k study requires explicit supersession; only the selected version is idempotent')
    replacement = None
    if supersede_provenance is not None:
        if data['example'] not in names:
            raise ValueError('Require a selected previous temporal study')
        replacement = manifest['cases'][names.index(data['example'])]
        if Path(replacement['source_provenance']).resolve() != Path(supersede_provenance).resolve(strict=True):
            raise ValueError('Expected previous study is not the selected version')
        previous = json.loads(Path(supersede_provenance).read_text())['data']
        validate_budget_extension(data, previous)
    if replace_provenance is not None:
        if collection != 'time_fusion' or data['example'] not in names:
            raise ValueError('Explicit replacements are limited to selected temporal figures')
        replacement = manifest['cases'][names.index(data['example'])]
        if Path(replacement['source_provenance']).resolve() != Path(replace_provenance).resolve(strict=True):
            raise ValueError('The expected previous provenance is not the selected version')
        previous = json.loads(Path(replace_provenance).read_text())['data']
        for key in ('steps', 'monitor_ids', 'test_ids', 'fields', 'budget'):
            if data[key] != previous[key]:
                raise ValueError('A figure replacement must preserve the comparison: ' + key)
        for new, old in zip(data['arms'], previous['arms'], strict=True):
            for key in ('method', 'run', 'checkpoint', 'monitor_errors', 'final_errors'):
                if new[key] != old[key]:
                    raise ValueError('A figure replacement cannot change training data: ' + key)
    existing_bytes = 0
    for case in manifest["cases"]:
        for figure in case["figures"]:
            for kind in ("png", "pdf"):
                size = (ROOT / figure[kind]).stat().st_size
                if size != figure["bytes"][kind]:
                    raise ValueError("Existing selected asset differs from its record")
                existing_bytes += size
    if existing_bytes != manifest["collection_bytes"]:
        raise ValueError("Existing collection byte count is inconsistent")

    keys = ("example", "fields", "budget", "seed", "steps", "monitor_ids", "test_ids",
            "reference", "convergence_metric", "final_metric", "checkpoint_selection",
            "smoothing", "uncertainty", "source_reports")
    record = {key: data[key] for key in keys}
    if "limitation" in data:
        record["limitation"] = data["limitation"]
    if 'convergence_axis' in data:
        record['convergence_axis'] = data['convergence_axis']
    if replacement is not None:
        record['supersedes'] = dict(source_provenance=replacement['source_provenance'],
                                    figures=replacement['figures'],
                                    reason=('User-requested paired cold15k controls supersede10k study'
                                            if supersede_provenance is not None else 'Cold wall-clock axis requested by user'))
    if suffix == "nonlocal_fem":
        for key in ("comparison_kind", "omitted_method"):
            record[key] = data[key]
    if suffix == "time_fusion":
        for key in ("comparison_kind", "pairing", "independent_local_single_case"):
            record[key] = data[key]
    record.update(source_provenance=str(source), reproduction=report["reproduction"],
                  fonts=report["fonts"], software=report["software"], arms=[], figures=[],
                  validation={"saved_field_norm_checks": len(data["saved_field_audit"]),
                              "visual_review": visual_review,
                              "pdf_checks": "Passed in linked renderer provenance"})
    for arm in data["arms"]:
        item = {key: arm[key] for key in ("method", "run", "configuration",
                                       "checkpoint", "monitor", "final_errors")}
        for key in ("run", "configuration", "checkpoint", "monitor"):
            if not Path(item[key]).is_absolute() or not Path(item[key]).exists():
                raise ValueError("Missing direct input: " + key)
        values = item["final_errors"]
        if (len(values) != tests or any(len(row) != len(data["fields"]) for row in values)
                or any(not math.isfinite(v) or v <= 0 for row in values for v in row)):
            raise ValueError("Invalid final field errors")
        item["field_means"] = {
            field["key"]: sum(row[index] for row in values) / tests
            for index, field in enumerate(data["fields"])
        }
        if suffix == "time_fusion":
            item["timing"] = arm["timing"]
            if 'elapsed_seconds' in arm:
                item['elapsed_seconds'] = arm['elapsed_seconds']
        record["arms"].append(item)

    copies = []
    added_bytes = 0
    for kind in ("convergence", "distribution"):
        entry = report["figures"][kind]
        figure = {"kind": kind, "bytes": {}, "size_mm": entry["size_mm"]}
        for extension in ("png", "pdf"):
            original = Path(entry[extension]).resolve(strict=True)
            name = f"{data['example']}_{suffix}_{kind}.{extension}"
            if original.name != name:
                raise ValueError("Unexpected rendered filename: " + str(original))
            target = ROOT / "assets/pino-studies" / name
            size = original.stat().st_size
            if size != entry["bytes"]["." + extension] or not 0 < size < 1024**2:
                raise ValueError("Rendered figure differs from its audit or exceeds budget")
            if target.exists() and target.read_bytes() != original.read_bytes() and replacement is None:
                raise ValueError("Refusing to overwrite a different selected figure: " + str(target))
            copies.append((original, target))
            figure[extension] = str(target.relative_to(ROOT))
            figure["bytes"][extension] = size
            added_bytes += size
        figure["png_dimensions"] = list(struct.unpack(">II", Path(entry["png"]).read_bytes()[16:24]))
        record["figures"].append(figure)

    if data["example"] in names and replacement is None:
        selected = manifest["cases"][names.index(data["example"])]
        if (selected["source_provenance"] != str(source)
                or selected["figures"] != record["figures"]
                or selected["arms"] != record["arms"]):
            raise ValueError("Selected case already exists; review a replacement explicitly")
        return {"status": "already_selected", "example": data["example"], "bytes": added_bytes}
    replaced_bytes = (sum(sum(f['bytes'].values()) for f in replacement['figures'])
                      if replacement is not None else 0)
    if existing_bytes - replaced_bytes + added_bytes >= manifest["maximum_collection_bytes"]:
        raise ValueError("Comparison collection would exceed its approved budget")
    for original, target in copies:
        if not target.exists() or replacement is not None:
            shutil.copyfile(original, target)
        if original.read_bytes() != target.read_bytes():
            raise ValueError("Copied asset is not byte-identical: " + str(target))
    if replacement is None:
        manifest["cases"].append(record)
    else:
        manifest['cases'][names.index(data['example'])] = record
    manifest["collection_bytes"] = existing_bytes - replaced_bytes + added_bytes
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(manifest_path)
    return {"status": "selected_locally", "example": data["example"], "bytes": added_bytes,
            "manifest": str(manifest_path), "deployment_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--visual-review", required=True)
    arguments = parser.parse_args()
    print(json.dumps(publish(arguments.provenance, arguments.visual_review), indent=2))
