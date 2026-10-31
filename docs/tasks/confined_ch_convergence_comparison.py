"""Compare frozen capacity-run histories at an explicit common update budget."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compare(first, second, end_step, tail_start, output, bank_growth=False, third=None):
    if not all(path.is_absolute() for path in (first, second, output)):
        raise ValueError("Explicit absolute snapshot and output paths required")
    if not 0 <= tail_start < end_step:
        raise ValueError("Require 0 <= tail_start < end_step")
    reports = [json.loads((path / "metrics.json").read_text()) for path in (first, second)]
    histories = [[json.loads(line) for line in (path / "metrics_snapshot.jsonl").read_text().splitlines()]
                 for path in (first, second)]
    cohort_keys = ("train_monitor_indices",) if bank_growth else (
        "train_monitor_indices", "validation_indices", "train_conditions")
    for key in cohort_keys:
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"Comparison cohorts differ: {key}")
    configs = [json.loads((Path(report["run"]) / "config.json").read_text()) for report in reports]
    for key in ("physical", "initial", "geometry", "nl", "sample", "loss", "reference"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Comparison numerical settings differ: {key}")
    geometry_mapping = None
    if bank_growth:
        if configs[0]["arch"] != configs[1]["arch"]:
            raise ValueError("Bank-size comparison requires identical architecture")
        if reports[0]["train_conditions"] >= reports[1]["train_conditions"]:
            raise ValueError("Second bank must have more training geometries")
        operators = [{key: value for key, value in config["operator"].items()
                      if key not in ("dataset_path", "train_conditions", "precompute")} for config in configs]
        precompute = [{key: value for key, value in config["operator"]["precompute"].items()
                       if key not in ("source_path", "reuse_mode")} for config in configs]
        if operators[0] != operators[1] or precompute[0] != precompute[1]:
            raise ValueError("Comparison differs beyond bank size/path/provenance")
        banks = [Path(config["operator"]["dataset_path"]) for config in configs]
        manifests = [json.loads((bank / "manifest.json").read_text()) for bank in banks]
        geometry_mapping = []
        for split, report_key in (("train", "train_monitor_indices"), ("validation", "validation_indices")):
            records = [[manifest["conditions"][index] for index in report[report_key]]
                       for manifest, report in zip(manifests, reports, strict=True)]
            if [row["draw_index"] for row in records[0]] != [row["draw_index"] for row in records[1]]:
                raise ValueError(f"Physical accepted geometry draws differ: {split}")
            for left, right in zip(*records, strict=True):
                with np.load(banks[0] / left["file"], allow_pickle=False) as a, np.load(
                        banks[1] / right["file"], allow_pickle=False) as b:
                    for key in ("points", "volumes", "holes", "time", "phi", "mu"):
                        np.testing.assert_array_equal(a[key], b[key])
                geometry_mapping.append(dict(split=split, draw_index=left["draw_index"],
                                             first_index=left["index"], second_index=right["index"]))
    elif configs[0]["operator"] != configs[1]["operator"]:
        raise ValueError("Comparison operator settings differ")
    train_settings = [{key: value for key, value in config["train"].items() if key != "iterations"}
                      for config in configs]
    if train_settings[0] != train_settings[1]:
        raise ValueError("Comparison optimizer, sampling seed or runtime settings differ")
    snapshot_paths = [first, second]
    if third is not None:
        if not bank_growth or not third.is_absolute():
            raise ValueError("Third capacity curve requires bank-growth mode and an absolute snapshot path")
        extra = json.loads((third / "metrics.json").read_text())
        extra_config = json.loads((Path(extra["run"]) / "config.json").read_text())
        expected = deepcopy(configs[0])
        if (extra_config["arch"]["width"] <= expected["arch"]["width"]
                or extra_config["arch"]["spatial_features"] <= expected["arch"]["spatial_features"]):
            raise ValueError("Third curve must increase backbone and Fourier width on the original bank")
        expected["arch"]["width"] = extra_config["arch"]["width"]
        expected["arch"]["spatial_features"] = extra_config["arch"]["spatial_features"]
        expected["train"]["iterations"] = extra_config["train"]["iterations"]
        expected["log"]["run_name"] = extra_config["log"]["run_name"]
        if expected != extra_config:
            raise ValueError("Third curve differs beyond declared capacity/budget and runtime name")
        for key in ("train_monitor_indices", "validation_indices", "train_conditions"):
            if reports[0][key] != extra[key]:
                raise ValueError(f"Third capacity curve cohort differs: {key}")
        reports.append(extra)
        configs.append(extra_config)
        histories.append([json.loads(line) for line in (third / "metrics_snapshot.jsonl").read_text().splitlines()])
        snapshot_paths.append(third)
    clipped = [[row for row in history if row["step"] <= end_step] for history in histories]
    if any(not rows or rows[-1]["step"] != end_step for rows in clipped):
        raise ValueError("All histories must include the declared common endpoint")
    if any([row["step"] for row in rows] != [row["step"] for row in clipped[0]] for rows in clipped[1:]):
        raise ValueError("Comparison evaluation steps differ")
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for rows, config, style in zip(clipped, configs, ("-", "--", ":")[:len(configs)], strict=True):
        for split, color, label in (("train", "#2171b5", f"Train monitors ({len(reports[0]['train_monitor_indices'])})"),
                                     ("test", "#d55e00", f"Validation ({len(reports[0]['validation_indices'])})")):
            x = np.asarray([row["step"] for row in rows]) / 1000
            y = np.asarray([row[f"{split}_phi_pooled"] for row in rows]) * 100
            if not np.all(np.isfinite(y)) or not np.all(y > 0):
                raise ValueError("Finite positive curve values required")
            if third is not None:
                curve_label = f"Width{config['arch']['width']}/N{config['operator']['train_conditions']} / {label}"
            else:
                curve_label = f"N{config['operator']['train_conditions']} / {label}" if bank_growth else f"Width{config['arch']['width']} / {label}"
            axes[0].semilogy(x, y, style, color=color, linewidth=1.7, label=curve_label)
            selected = x >= tail_start / 1000
            axes[1].plot(x[selected], y[selected], style, color=color, linewidth=1.7, label=curve_label)
    axes[0].set(title="(a) Full common history", ylabel=r"Phase relative $L^2$ error (%)")
    axes[1].set(title="(b) Late common history (linear scale)",
                ylabel=r"Phase relative $L^2$ error (%)", xlim=(tail_start / 1000, end_step / 1000))
    for ax in axes:
        ax.set_xlabel(f"{configs[0]['train']['optimizer'].upper()} updates (thousands)")
        ax.grid(alpha=.18, which="both")
    axes[0].legend(frameon=False, fontsize=8)
    identity = f"Width{configs[0]['arch']['width']} | N{reports[0]['train_conditions']} vs N{reports[1]['train_conditions']}" if bank_growth else f"Width{configs[0]['arch']['width']} vs width{configs[1]['arch']['width']}"
    if third is not None:
        identity = " / ".join(f"W{config['arch']['width']},N{config['operator']['train_conditions']}" for config in configs)
    fig.suptitle(f"{identity} | common {end_step:,} updates | native nonlocal reference\n"
                 "Five stored times pooled; unsmoothed L2; same physical training-monitor and validation geometries",
                 fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"comparison.{suffix}", dpi=210)
    plt.close(fig)
    metrics = {
        "snapshots": [str(path) for path in snapshot_paths], "end_step": end_step,
        "tail_start_step": tail_start, "matched_evaluation_records": len(clipped[0]),
        "bank_growth": bank_growth, "physical_geometry_mapping": geometry_mapping,
        "third_capacity_curve": third is not None,
        "endpoints": [{"width": config["arch"]["width"], "training_geometries": config["operator"]["train_conditions"], **{key: rows[-1][key] for key in
                      ("step", "train_phi_pooled", "test_phi_pooled", "train_phi_terminal", "test_phi_terminal")}}
                      for rows, config in zip(clipped, configs, strict=True)],
        "scope": ("Recorded common-budget bank-size comparison; identical architecture/settings and physical monitor/reference arrays. "
                 "Training-ID samples differ with expanded support. One seed/eight reused validation domains; only16training monitors, no final50kclaim."
                 if bank_growth else "Recorded common-budget curves; one runtime seed, eight reused development geometries. "
                 "Different-width fresh initialization, joint Fourier/backbone change. "
                 "Training means only16monitors out of512inputs; not a fresh test or final50kcomparison."),
    }
    if third is not None:
        metrics["scope"] = ("Three recorded histories at a common update budget on the same16training monitors/8reserved physical geometries. "
                            "First-to-second changes bank size with the same architecture; first-to-third changes backbone/Fourier width on the original512bank. "
                            "Third curve has different-shaped fresh weights and uses only the historical prefix of a user-stopped run. "
                            "One seed/reused development cohort, raw native L2, not final50k or a seed-stability claim.")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--tail-start-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-growth", action="store_true",
                        help="Compare expanded banks with exact accepted-draw/reference checks")
    parser.add_argument("--third", type=Path,
                        help="Optional wider-model snapshot on the first bank; common steps only")
    args = parser.parse_args()
    compare(args.first, args.second, args.end_step, args.tail_start_step, args.output, args.bank_growth, args.third)
