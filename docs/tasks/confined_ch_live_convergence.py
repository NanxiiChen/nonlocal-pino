"""Freeze complete live records and plot training-monitor/validation convergence."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_complete(path):
    lines = path.read_bytes().splitlines(keepends=True)
    records = [json.loads(line) for line in lines if line.endswith(b"\n")]
    if not records or any(b["step"] <= a["step"] for a, b in zip(records, records[1:])):
        raise ValueError(f"Nonempty, strictly ordered complete records required: {path}")
    return records


def report(run, output, tail_start, median_window):
    if not run.is_absolute() or not output.is_absolute():
        raise ValueError("Use explicit absolute run and output paths")
    if tail_start < 0 or median_window < 1 or median_window % 2 != 1:
        raise ValueError("Nonnegative tail start and positive odd median window required")
    config = json.loads((run / "config.json").read_text())
    if "continuation" in config["train"]:
        raise ValueError("This live snapshot reader requires a cold-start run")
    optimizer = config["train"]["optimizer"].upper()
    precision = {"float32": "FP32", "float64": "FP64"}[config["train"]["compute_dtype"]]
    history = _read_complete(run / "metrics.jsonl")
    latest_step = history[-1]["step"]
    if latest_step > config["train"]["iterations"]:
        raise ValueError("Recorded step exceeds the declared training budget")
    budget_complete = latest_step == config["train"]["iterations"]
    loss_all = _read_complete(run / "loss.jsonl")
    loss = [row for row in loss_all if row["step"] <= latest_step]
    train_ids = history[0]["train_monitor_indices"]
    validation_ids = history[0]["test_indices"]
    if set(train_ids) & set(validation_ids):
        raise ValueError("Training-monitor and validation geometry IDs overlap")
    keys = tuple(f"{split}_{field}" for split in ("train", "test")
                 for field in ("phi_pooled", "phi_terminal", "mu_pooled", "mu_terminal"))
    for row in history:
        if row["train_monitor_indices"] != train_ids or row["test_indices"] != validation_ids:
            raise ValueError(f"Monitoring cohort changes at step {row['step']}")
        by_id = {item["index"]: item for item in row["conditions"]}
        if len(by_id) != len(row["conditions"]) or set(by_id) != set(train_ids + validation_ids):
            raise ValueError(f"Incomplete or duplicate geometry records at step {row['step']}")
        for split, indices in (("train", train_ids), ("test", validation_ids)):
            for field in ("phi_pooled", "phi_terminal", "mu_pooled", "mu_terminal"):
                expected = np.mean([by_id[index][field] for index in indices])
                if not np.isclose(row[f"{split}_{field}"], expected, rtol=0, atol=5e-16):
                    raise ValueError(f"Aggregate mismatch at step {row['step']}: {split}/{field}")
    values = np.asarray([[row[key] for key in keys] for row in history])
    loss_values = np.asarray([row["loss"] for row in loss])
    if not np.all(np.isfinite(values)) or not np.all(values > 0):
        raise ValueError("Finite positive error records required for log axes")
    if len(loss) < median_window or not np.all(np.isfinite(loss_values)) or not np.all(loss_values > 0):
        raise ValueError("Sufficient finite positive loss records required")
    late = [row for row in history if row["step"] >= tail_start]
    if len(late) < 3:
        raise ValueError("At least three late evaluation records required")

    compact = [{key: row[key] for key in ("step", "elapsed_seconds", *keys)} for row in history]
    milestones = [row for row in compact if row["step"] in (0, 10000, 20000, 30000)
                  or row["step"] == latest_step]
    tail = [row for row in history if row["step"] >= latest_step - 5000]
    x_tail = np.asarray([row["step"] for row in tail], dtype=float) / 1000
    trends = {}
    for key in ("train_phi_pooled", "test_phi_pooled", "train_phi_terminal", "test_phi_terminal"):
        y = np.asarray([row[key] for row in tail]) * 100
        trends[key] = {
            "start_step": tail[0]["step"], "end_step": latest_step,
            "slope_percentage_points_per1000updates": float(np.polyfit(x_tail, y, 1)[0]),
            "start_percent": float(y[0]), "end_percent": float(y[-1]),
        }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "output": str(output), "snapshot_step": latest_step,
        "requested_steps": config["train"]["iterations"],
        "budget_complete": budget_complete,
        "latest_logged_loss_step": loss_all[-1]["step"],
        "train_monitor_indices": train_ids, "validation_indices": validation_ids,
        "train_conditions": config["operator"]["train_conditions"],
        "evaluation_records": len(history), "loss_records_plotted": len(loss),
        "tail_start_step": tail_start, "loss_median_window_records": median_window,
        "milestones": milestones, "last5000_trends": trends,
        "latest_phase_gap_percentage_points": float(100 * (
            history[-1]["test_phi_pooled"] - history[-1]["train_phi_pooled"])),
        "aggregate_reconstruction": "passed for every split/field/evaluation",
        "l2_smoothing": "none; lines connect original evaluation points",
        "scope": "Mean per-geometry relative L2 against unchanged native nonlocal reference. "
                 "Pooled fields use five stored times, terminal uses physical T. "
                 "Only the fixed training-monitor subset, not the whole training bank. "
                 "Eight reused development geometries; one seed, recorded checkpoint history, not a fresh test. "
                 "Reference fields never enter the PDE-only optimizer loss.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, rows in (("metrics_snapshot.jsonl", history), ("loss_snapshot.jsonl", loss)):
        (output / name).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.1), layout="constrained")
    x = np.asarray([row["step"] for row in history]) / 1000
    colors = {"train": "#2171b5", "test": "#d55e00"}
    labels = {"train": f"Train monitors ({len(train_ids)}/{result['train_conditions']} geometries)",
              "test": f"Validation ({len(validation_ids)} geometries)"}
    for ax, field, title in ((axes[0, 0], "phi_pooled", "(a) Phase error: five stored times pooled"),
                             (axes[0, 1], "phi_terminal", f"(b) Phase error at t = {config['physical']['terminal_time']:g}")):
        for split in ("train", "test"):
            y = np.asarray([row[f"{split}_{field}"] for row in history]) * 100
            ax.semilogy(x, y, color=colors[split], linewidth=1.7, label=labels[split])
            ax.scatter(x[-1], y[-1], color=colors[split], s=22, zorder=3)
            ax.annotate(f"{y[-1]:.2f}%", (x[-1], y[-1]), xytext=(7, 0),
                        textcoords="offset points", va="center", color=colors[split], fontweight="bold")
        ax.axhline(10, color="#999999", linestyle=":", linewidth=1)
        ax.set(title=title, xlabel=f"{optimizer} updates (thousands)", ylabel=r"Relative $L^2$ error (%)",
               xlim=(0, x[-1] + max(3, x[-1] * .08)))
        ax.grid(alpha=.18, which="both")
    axes[0, 0].legend(loc="upper right", fontsize=9, frameon=False)

    ax = axes[1, 0]
    late_x = np.asarray([row["step"] for row in late]) / 1000
    for split in ("train", "test"):
        y = np.asarray([row[f"{split}_phi_pooled"] for row in late]) * 100
        ax.plot(late_x, y, color=colors[split], linewidth=1.7, marker="o", markersize=2.5)
        ax.annotate(f"{y[-1]:.2f}%", (late_x[-1], y[-1]), xytext=(7, 0),
                    textcoords="offset points", va="center", color=colors[split], fontweight="bold")
    ax.set(title="(c) Late-stage pooled phase error (linear scale)",
           xlabel=f"{optimizer} updates (thousands)", ylabel=r"Relative $L^2$ error (%)",
           xlim=(tail_start / 1000, x[-1] + max(2, (x[-1] - tail_start / 1000) * .1)))
    ax.grid(alpha=.18)

    ax = axes[1, 1]
    loss_x = np.asarray([row["step"] for row in loss]) / 1000
    half = median_window // 2
    rolling = np.median(np.lib.stride_tricks.sliding_window_view(loss_values, median_window), axis=1)
    ax.semilogy(loss_x, loss_values, color="#b3b3b3", linewidth=.8, alpha=.7, label="Raw sampled loss")
    ax.semilogy(loss_x[half:len(loss_x) - half] if half else loss_x, rolling,
                color="#333333", linewidth=1.5, label=f"{median_window}-record median (display only)")
    ax.set(title="(d) Physics-only optimizer loss", xlabel=f"{optimizer} updates (thousands)",
           ylabel="Sampled preconditioned PDE loss")
    ax.grid(alpha=.18, which="both")
    ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        if latest_step >= 30000:
            ax.axvline(30, color="#aaaaaa", linestyle="--", linewidth=.8)
    endpoint = "Fixed-budget endpoint" if budget_complete else "Live snapshot"
    fig.suptitle(f"Confined CH geometry operator | width {config['arch']['width']} | {optimizer} {precision}\n"
                 f"{endpoint}: {latest_step:,}/{result['requested_steps']:,} updates; native nonlocal reference",
                 fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"convergence.{suffix}", dpi=210)
    plt.close(fig)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-start-step", type=int, required=True)
    parser.add_argument("--loss-median-window", type=int, required=True)
    args = parser.parse_args()
    report(args.run, args.output, args.tail_start_step, args.loss_median_window)
