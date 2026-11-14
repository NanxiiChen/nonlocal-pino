"""Reproduce the channel FEM/nonlocal figures from saved logs and fields on CPU."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from fontTools.ttLib import TTFont
import fitz
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde
from local_trend import smooth_interval


FIELDS = (
    ("phi", "Phase", slice(0, 1), 0),
    ("velocity", "Velocity", slice(2, 4), 1),
    ("pressure", "Pressure", slice(4, 5), 1),
    ("chemical", "Chemical potential", slice(1, 2), 1),
)
METHODS = (
    ("nonlocal", "Nonlocal PINO", "#E64B35", "-", "o"),
    ("fem", "FEM PINO", "#00A087", "-", "s"),
)
SIZE_MM = (170, 54)
SMOOTHING = dict(fraction=.22, block_size=4, replicates=2000, seed=20261114, grid_points=401)


def read(path):
    return json.loads(Path(path).read_text())


def load(comparison):
    report = read(comparison)
    if report["status"] != "passed" or report["paired"]["updates"] != 20000:
        raise ValueError("Require the completed, audited 20k comparison")
    result = []
    reference_paths = None
    for method, label, color, style, marker in METHODS:
        run = Path(report["historical"] if method == "nonlocal" else report["fem_run"])
        cohort = read(run / "unseen3/evaluation.json")
        config = read(run / "config.json")
        if config["train"]["seed"] != 42 or config["train"]["iterations"] != 20000:
            raise ValueError(f"Unexpected seed or budget: {run}")
        checkpoint = (run / "model_final.eqx").resolve(strict=True)
        records = cohort["per_geometry"]
        if [x["index"] for x in records] != [0, 1, 2]:
            raise ValueError("Use every predeclared reserved geometry in order")
        paths = [x["reference"] for x in records]
        if reference_paths is not None and paths != reference_paths:
            raise ValueError("Methods must share reference cases")
        reference_paths = paths
        samples = []
        for case in records:
            prediction = run / "unseen3" / f"case_{case['index']:03d}" / "prediction.npz"
            with np.load(prediction, allow_pickle=False) as saved, np.load(case["reference"], allow_pickle=False) as ref:
                for key in ("points", "time", "amplitudes"):
                    np.testing.assert_array_equal(saved[key], ref[key])
                np.testing.assert_array_equal(saved["reference"], ref["fields"])
                if saved["fields"].shape != (13, 3840, 5):
                    raise ValueError("Unexpected saved field shape")
                errors = {}
                for key, _, channel, begin in FIELDS:
                    actual = np.asarray(saved["fields"][begin:, :, channel], dtype=np.float64)
                    target = np.asarray(ref["fields"][begin:, :, channel], dtype=np.float64)
                    denominator = float(np.linalg.norm(target))
                    if denominator <= 0 or not np.isfinite(denominator):
                        raise ValueError(f"Invalid relative-norm denominator: {key}")
                    error = float(np.linalg.norm(actual - target) / denominator)
                    if not np.isfinite(error) or error <= 0:
                        raise ValueError(f"Log axes require positive finite error: {key}")
                    np.testing.assert_allclose(error, case["metrics"][key], rtol=0, atol=1e-12)
                    errors[key] = error
                samples.append(dict(index=case["index"], prediction=str(prediction),
                                    reference=case["reference"], time=saved["time"].tolist(),
                                    points=3840, errors=errors))
        for key, *_ in FIELDS:
            np.testing.assert_allclose(np.mean([x["errors"][key] for x in samples]),
                                       cohort["aggregate"][key]["mean"], rtol=0, atol=1e-12)
        logs = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
        if [x["step"] for x in logs] != list(range(0, 20001, 500)):
            raise ValueError("Require the actual 41 monitoring records, including initialization")
        for row in logs:
            if any(not np.isfinite(row[key]) or row[key] <= 0 for key, *_ in FIELDS):
                raise ValueError("Invalid monitoring error")
        result.append(dict(method=method, label=label, color=color, linestyle=style, marker=marker,
                           run=str(run), checkpoint=str(checkpoint), config=str(run / "config.json"),
                           monitor=str(run / "metrics.jsonl"), evaluation=str(run / "unseen3/evaluation.json"),
                           seed=42, checkpoint_step=20000, samples=samples,
                           development_reference=logs[0]["reference"],
                           convergence=[dict(step=x["step"], **{key:x[key] for key, *_ in FIELDS}) for x in logs]))
        if any(x["reference"] != logs[0]["reference"] for x in logs):
            raise ValueError("Monitoring reference changed during training")
    if result[0]["development_reference"] != result[1]["development_reference"]:
        raise ValueError("Development references differ")
    return result


def style(font):
    with TTFont(font) as face:
        if face["name"].getDebugName(1) != "Helvetica" or "glyf" not in face:
            raise ValueError("Expected an actual Helvetica TrueType font")
        if face["OS/2"].fsType & (0x0002 | 0x0100 | 0x0200):
            raise ValueError("Font must permit embedding and subsetting")
    font_manager.fontManager.addfont(str(font))
    if Path(font_manager.findfont("Helvetica", fallback_to_default=False)).resolve() != font:
        raise ValueError("Unexpected font resolution")
    return {
        "font.family":"sans-serif", "font.sans-serif":["Helvetica"], "font.size":6,
        "axes.labelsize":7, "axes.titlesize":7, "xtick.labelsize":6, "ytick.labelsize":6,
        "legend.fontsize":6, "legend.frameon":False, "legend.labelcolor":"#222222",
        "text.color":"#222222", "axes.labelcolor":"#222222", "axes.titlecolor":"#222222",
        "xtick.color":"#222222", "ytick.color":"#222222", "axes.linewidth":.5,
        "xtick.major.width":.5, "ytick.major.width":.5, "xtick.minor.width":.5,
        "ytick.minor.width":.5, "xtick.direction":"in", "ytick.direction":"in",
        "pdf.fonttype":42, "ps.fonttype":42, "pdf.use14corefonts":False,
        "text.usetex":False, "mathtext.fontset":"dejavusans", "mathtext.fallback":None,
        "figure.facecolor":"white", "axes.facecolor":"white", "savefig.facecolor":"white",
    }


def attach_cohort(arms, path, comparison):
    """Audit every displayed checkpoint norm and derive statistics from cases."""
    data = read(path)
    if (data["schema"] != "channel_fem_checkpoint_cohort_v1" or data["status"] != "passed"
            or data["steps"] != [0, 5000, 10000, 15000, 20000]
            or data["samples"] != 3 or data["std_ddof"] != 1
            or Path(data["comparison"]).resolve() != comparison.resolve()):
        raise ValueError("Require the actual five-checkpoint, three-case cohort evaluation")
    for arm, measured in zip(arms, data["arms"], strict=True):
        if arm["method"] != measured["method"] or arm["run"] != measured["run"]:
            raise ValueError("Cohort method/run mismatch")
        reference_paths = [x["reference"] for x in arm["samples"]]
        if [x["step"] for x in measured["convergence"]] != data["steps"]:
            raise ValueError("Missing actual checkpoint")
        for row in measured["convergence"]:
            if [x["index"] for x in row["samples"]] != [0, 1, 2]:
                raise ValueError("Incomplete or reordered cohort")
            for case, reference in zip(row["samples"], reference_paths, strict=True):
                if case["reference"] != reference or not Path(case["checkpoint"]).is_file():
                    raise ValueError("Reference/checkpoint provenance mismatch")
                with np.load(case["prediction"], allow_pickle=False) as saved, np.load(reference, allow_pickle=False) as ref:
                    for key in ("points", "time", "amplitudes"):
                        np.testing.assert_array_equal(saved[key], ref[key])
                    for key, _, channel, begin in FIELDS:
                        a, b = saved["fields"][begin:, :, channel], ref["fields"][begin:, :, channel]
                        norm = float(np.linalg.norm(a-b)/np.linalg.norm(b))
                        np.testing.assert_allclose(norm, case["errors"][key], rtol=0, atol=1e-12)
            for key, *_ in FIELDS:
                values = np.array([x["errors"][key] for x in row["samples"]])
                mean, std = values.mean(), values.std(ddof=1)
                np.testing.assert_allclose([mean, std], [row["aggregate"][key]["mean"],
                                                        row["aggregate"][key]["std"]], rtol=0, atol=1e-12)
                if not np.isfinite(values).all() or mean-std <= 0:
                    raise ValueError("Mean +/- sample SD must be finite and positive for log axes")
        arm["historical_final_samples"] = arm["samples"]
        arm["cohort_convergence"] = measured["convergence"]
        arm["samples"] = measured["convergence"][-1]["samples"]
        arm["checkpoint"] = arm["samples"][0]["checkpoint"]
        arm["cohort_evaluation"] = str(path.resolve())
    return data


def panels():
    fig, axes = plt.subplots(1, 4, figsize=np.asarray(SIZE_MM)/25.4)
    fig.subplots_adjust(left=.085, right=.99, bottom=.22, top=.73, wspace=.39)
    for ax, (_, title, _, _) in zip(axes, FIELDS, strict=True):
        ax.set_title(title, pad=5)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _:f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="major", length=2.5)
        ax.tick_params(which="minor", length=1.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Rel. L² error", labelpad=5)
    return fig, axes


def save(fig, axes, destination, font, pdffonts, output):
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            text.set_fontproperties(font_manager.FontProperties(fname=str(font), size=text.get_fontsize()))
            text.set_color("#222222")
    fig.savefig(destination.with_suffix(".pdf"), metadata={"CreationDate":None, "ModDate":None})
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    limits = [dict(autoscale_y=ax.get_autoscaley_on(), y_limits=list(ax.get_ylim())) for ax in axes]
    if not all(x["autoscale_y"] for x in limits):
        raise ValueError("Do not set manual y limits")
    plt.close(fig)
    with fitz.open(destination.with_suffix(".pdf")) as doc:
        page = doc[0]
        np.testing.assert_allclose([page.rect.width, page.rect.height], np.asarray(SIZE_MM)*72/25.4,
                                   atol=1e-4, rtol=0)
        spans = [s for block in page.get_text("dict")["blocks"] if block["type"] == 0
                 for line in block["lines"] for s in line["spans"]]
        if not spans or any(not 4.99 <= s["size"] <= 7.01 for s in spans):
            raise ValueError("Actual PDF text must be 5–7 pt")
        if page.get_images() or any(not page.rect.contains(fitz.Rect(s["bbox"])) for s in spans):
            raise ValueError("Figure must be vector and all text inside the page")
        for font_row in page.get_fonts(full=True):
            if font_row[2] not in ("TrueType", "Type0") or not doc.extract_font(font_row[0])[3]:
                raise ValueError("Non-embedded or unsupported PDF font")
        edit = fitz.open(stream=doc.tobytes(), filetype="pdf")
        for index, (old, new) in enumerate((("Rel. L² error", "Rel. L² score"), ("Pressure", "Pressure field"))):
            matches = edit[0].search_for(old)
            if not matches:
                raise ValueError(f"PDF editable label missing: {old}")
            edit[0].add_redact_annot(matches[0], fill=(1, 1, 1))
            edit[0].apply_redactions(images=0, graphics=0)
            edit[0].insert_text((30, 14+9*index), new, fontsize=7, fontname="plot_helvetica", fontfile=str(font))
            if new not in edit[0].get_text():
                raise ValueError("Independent text-edit probe failed")
        probe = output / (destination.name+"_edit_probe.pdf")
        edit.save(probe)
        edit.close()
        with Image.open(destination.with_suffix(".png")) as preview:
            pixels = list(preview.size)
        result = dict(pdf=str(destination.with_suffix(".pdf")), png=str(destination.with_suffix(".png")),
                      size_mm=SIZE_MM, png_pixels=pixels,
                      bytes={ext:destination.with_suffix("."+ext).stat().st_size for ext in ("png", "pdf")},
                      axes=limits, font_sizes_pt=sorted({round(x["size"], 3) for x in spans}),
                      all_text_dark=all(x["color"] == 0x222222 for x in spans),
                      editable_label_probe=str(probe),
                      pdffonts=subprocess.check_output([str(pdffonts), str(destination.with_suffix(".pdf"))],text=True))
    return result


def render(arms, output, font, pdffonts):
    fig, axes = panels()
    trend_records = []
    for ax, (key, *_ ) in zip(axes, FIELDS, strict=True):
        for arm in arms:
            x = [row["step"] for row in arm["convergence"]]
            y = np.array([row[key] for row in arm["convergence"]])
            fitted = smooth_interval(x, y, **SMOOTHING)
            line, = ax.plot(x, y, color=arm["color"], linestyle="-", lw=.55, alpha=.35, zorder=2)
            np.testing.assert_array_equal(line.get_ydata(), y)
            ax.fill_between(fitted["x"], fitted["lower"], fitted["upper"],
                            color=arm["color"], alpha=.18, linewidth=0, zorder=1)
            ax.plot(fitted["x"], fitted["smooth"], color=arm["color"], linestyle="-",
                    lw=1.05, label=arm["label"], zorder=3)
            payload = output/f"{arm['method']}_{key}_local_trend.npz"
            np.savez_compressed(payload, raw_steps=np.asarray(x), raw_values=y, **fitted)
            trend_records.append(dict(method=arm["method"], field=key, data=str(payload),
                                      step_values=np.asarray(x).tolist(), raw_values=y.tolist()))
        ax.set_xticks([0, 5000, 10000, 15000, 20000], ["0", "5k", "10k", "15k", "20k"])
        ax.set_xlabel("Step", labelpad=4)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(.54, 1), handlelength=2.8)
    convergence = save(fig, axes, output/"channel_nsch2d_fem_convergence", font, pdffonts, output)
    convergence["trend_data"] = trend_records
    convergence["smoothing"] = SMOOTHING

    fig, axes = panels()
    fig.subplots_adjust(bottom=.19, top=.82)
    densities = []
    for ax, (key, *_) in zip(axes, FIELDS, strict=True):
        for index, arm in enumerate(arms):
            values = np.array([x["errors"][key] for x in arm["samples"]])
            log_values = np.log10(values)
            kde = gaussian_kde(log_values, bw_method="scott")
            bandwidth = float(np.sqrt(kde.covariance[0, 0]))
            grid = np.linspace(log_values.min()-3*bandwidth, log_values.max()+3*bandwidth, 256)
            density = kde(grid)
            half_width = .32*density/density.max()
            ax.fill_betweenx(10**grid, index-half_width, index+half_width, color=arm["color"],
                             alpha=.85, edgecolor="#333333", linewidth=.6)
            quartiles = np.quantile(values, [.25, .5, .75])
            for q in quartiles:
                width = .32*float(kde(np.log10(q))[0])/density.max()
                ax.plot([index-width, index+width], [q, q], color="#333333", lw=.6, linestyle=(0, (2, 1.5)))
            densities.append(dict(method=arm["method"], field=key, bandwidth_log10=bandwidth,
                                  scott_factor=float(kde.factor), quartiles=quartiles.tolist(),
                                  grid_points=256, tail_bandwidths=3, maximum_half_width=.32))
        ax.set_xticks([0, 1], ["Nonlocal", "FEM PINO"])
        ax.margins(x=.12)
    distribution = save(fig, axes, output/"channel_nsch2d_fem_distribution", font, pdffonts, output)
    distribution["kde"] = densities
    return dict(convergence=convergence, distribution=distribution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--cohort-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--pdffonts", type=Path, required=True)
    args = parser.parse_args()
    font, pdffonts = args.font.resolve(strict=True), args.pdffonts.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arms = load(args.comparison)
    cohort = attach_cohort(arms, args.cohort_evaluation, args.comparison)
    with plt.rc_context(style(font)):
        figures = render(arms, output, font, pdffonts)
    provenance = dict(schema="channel_fem_comparison_figures_v3", comparison=str(args.comparison.resolve()),
                      cohort_evaluation=str(args.cohort_evaluation.resolve()),
                      renderer=str(Path(__file__).resolve()),
                      smoothing_source=str(Path(__file__).with_name("local_trend.py").resolve()),
                      arms=arms, figures=figures,
                      reproduction=dict(
                          render_argv=[str(Path(sys.executable).absolute()), str(Path(__file__).resolve()),
                                       "--comparison", str(args.comparison.resolve()),
                                       "--cohort-evaluation", str(args.cohort_evaluation.resolve()),
                                       "--output", str(output), "--font", str(font), "--pdffonts", str(pdffonts)],
                          evaluate_argv=[str(Path(sys.executable).absolute()), "-m", "scripts.evaluate_channel_fem_checkpoints",
                                         "--comparison", str(args.comparison.resolve()), "--output", str(args.cohort_evaluation.resolve().parent)],
                          environment=dict(PYTHONPATH="/tmp/pino-pdf-audit", OPENBLAS_NUM_THREADS="1",
                                           OMP_NUM_THREADS="1", XLA_FLAGS="--xla_gpu_autotune_level=0"),
                          cwd=str(Path(__file__).resolve().parents[1]),
                          note="Renderer reuses saved inference on CPU. For a fresh GPU reevaluation choose a new output directory, then pass its evaluation.json to the renderer; existing inference outputs are protected."),
                      metrics="Unweighted discrete space-time L2 per field and geometry; velocity uses the joint vector norm. Phase uses all 13 saved times, velocity/pressure/mu use 12 positive times. No channel averaging in these field-wise panels.",
                      distribution="All three reserved geometries, final20k; Gaussian KDE in log10-error with Scott bandwidth, three-bandwidth tails, each violin normalized to equal maximum width, internal25/50/75percentiles. N=3 is a limited descriptive visualization, not population-density evidence; tails are not additional observations.",
                      convergence="41 actual records per method on one development geometry. Thin solid raw traces; thick solid local-linear LOWESS in log10error, fraction0.22(9neighbors), robust iterations0. Shading: approximate pointwise95%basic circular-block residual-bootstrap trend interval, block4records,2000replicates, seed20261114.401display locations are fitted values, not new observations. Assumes a smooth trend and approximately stationary short-memory log residuals; no cross-seed/cohort uncertainty, simultaneous coverage or correction for unknown smoothing bias.",
                      smoothing=SMOOTHING,
                      statistical_references=[
                          "https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html",
                          "https://www.statsmodels.org/stable/examples/notebooks/generated/lowess.html",
                          "https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html",
                          "https://arch.readthedocs.io/en/latest/bootstrap/confidence-intervals.html"],
                      font=str(font), font_source="https://raw.githubusercontent.com/NanxiiChen/PI-CViT/main/helvetica/Helvetica.ttf",
                      mathtext="dejavusans, fallback=None; these labels use editable Helvetica Unicode L²",
                      versions=dict(python=platform.python_version(), matplotlib=matplotlib.__version__, numpy=np.__version__, pymupdf=fitz.VersionBind),
                      validations=dict(saved_field_reductions=144, evaluated_checkpoints=5, paired_cases=3,
                                       actual_development_values=328,
                                       final_replay=[arm["final_replay"] for arm in cohort["arms"]],
                                       pdf_text_edit="Programmatic edit of temporary PDF copies; no GUI editor claim"))
    (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps({"status":"passed", "output":str(output), "figures":figures},indent=2))


if __name__ == "__main__":
    main()
