"""Plot local FEM, direct nonlocal, and optional PINN stress-strain curves."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load_curve(path):
    """Load average strain and reaction stress from a result archive."""
    data = np.load(path)
    for name in ("u_bar", "reaction_star"):
        if name not in data.files:
            raise KeyError(f"{path} is missing {name}")
    strain = np.asarray(data["u_bar"], dtype=np.float64)
    stress = np.asarray(data["reaction_star"], dtype=np.float64)
    if strain.ndim != 1 or stress.shape != strain.shape:
        raise ValueError(f"Invalid stress-strain arrays in {path}")
    if not np.all(np.isfinite(strain)) or not np.all(np.isfinite(stress)):
        raise FloatingPointError(f"Non-finite stress-strain data in {path}")
    return strain, stress


def _load_tensorboard_curve(event_dir, du):
    """Load the old exploratory PINN reaction scalar from TensorBoard."""
    event_files = sorted(Path(event_dir).glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise ValueError(
            f"Expected exactly one TensorBoard event file in {event_dir}, "
            f"found {len(event_files)}"
        )
    accumulator = EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    tag = "train/reaction_star"
    if tag not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"TensorBoard event file is missing scalar {tag}")
    events = accumulator.Scalars(tag)
    if not events:
        raise ValueError(f"TensorBoard scalar {tag} is empty")
    steps = np.asarray([event.step for event in events], dtype=np.float64)
    stress = np.asarray([event.value for event in events], dtype=np.float64)
    strain = (steps + 1.0) * float(du)
    return strain, stress


def plot_comparison(local_path, baseline_path, pinn_events, pinn_du, output_path):
    """Create a full-path and localization zoom comparison figure."""
    local_strain, local_stress = _load_curve(local_path)
    baseline_strain, baseline_stress = _load_curve(baseline_path)
    pinn_strain, pinn_stress = _load_tensorboard_curve(pinn_events, pinn_du)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    axes[0].plot(local_strain, local_stress, color="#1f77b4", linewidth=1.4,
                 label="local FEM reference")
    axes[0].plot(baseline_strain, baseline_stress, color="#d62728", linewidth=1.4,
                 label="direct nonlocal baseline")
    axes[0].plot(pinn_strain, pinn_stress, color="#2ca02c", linewidth=1.3,
                 linestyle="--", label="PINN exploratory (pre-fix)")
    axes[0].set_xlabel(r"average strain $\bar{\varepsilon}=\bar{u}/L_0$")
    axes[0].set_ylabel(r"reaction stress $\sigma^*$")
    axes[0].set_title("Full loading path")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    zoom_min = 0.0112
    zoom_max = 0.0120
    for strain, stress, color, linestyle, label in (
        (local_strain, local_stress, "#1f77b4", "-", "local FEM reference"),
        (baseline_strain, baseline_stress, "#d62728", "-", "direct nonlocal baseline"),
        (pinn_strain, pinn_stress, "#2ca02c", "--", "PINN exploratory (pre-fix)"),
    ):
        mask = (strain >= zoom_min) & (strain <= zoom_max)
        axes[1].plot(strain[mask], stress[mask], color=color,
                     linestyle=linestyle, linewidth=1.5, label=label)
    axes[1].axvline(0.01175, color="black", linestyle=":", linewidth=1.0,
                    label="baseline localization snapshot")
    axes[1].set_xlim(zoom_min, zoom_max)
    axes[1].set_xlabel(r"average strain $\bar{\varepsilon}=\bar{u}/L_0$")
    axes[1].set_ylabel(r"reaction stress $\sigma^*$")
    axes[1].set_title("Localization zoom")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    figure.suptitle(
        "Fracture1D stress-strain comparison; PINN trace is diagnostic only",
        fontsize=12,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--pinn-events", required=True)
    parser.add_argument("--pinn-du", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_comparison(
        args.local,
        args.baseline,
        args.pinn_events,
        args.pinn_du,
        args.output,
    )


if __name__ == "__main__":
    _main()
