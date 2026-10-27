# Fresh two-phase inverse-cubic map ablation

Goal: test whether removing the inverse-cubic output coordinate materially
worsens the current recommended prescribed-vortex CH result.

Acceptance: serial matched fresh additive and inverse-cubic nonlocal cold starts
through train.py, identical raw parameters/config except phase map and run name,
same exact phase IC/mass, spectral conservative flux, original two PDE losses,
FP64/ordinary, SOAP5000+L-BFGS1000. Independently replay final checkpoints and
report common-local/native pooled, terminal/interface errors and timings.

Context: soft mass penalty was just removed under hard projection. Older cubic
results have known same-seed variability, so use a fresh paired control rather
than only a historical favorable checkpoint. One pair is not a robustness
study; SOAP stability remains explicitly deferred. No default or Git operation
requested. Reference/checkpoint paths are retained.

Status: complete (2026-10-27). Both trainers and CPU analyzer exit0.
Blocker: none.
Next action: none. Keep existing defaults and both explicit map options.

Run root: /root/autodl-tmp/tf-logs/twophase2d/.
New runs: 20261027-vortex-map-ablation-additive and
20261027-vortex-map-ablation-cubic. Full reference remains
/root/autodl-tmp/data/twophase2d/results_periodic_vortex.npz.

Both final checkpoints are at6000 updates (SOAP5000 + L-BFGS1000). L-BFGS
ends at the iteration budget, not the requested tolerance, in both. No
reference fields enter the training loss, and no best-checkpoint search occurs.

| Final metric | Additive/direct | Inverse cubic |
|---|---:|---:|
| Common-local pooled phi L2 | 3.200643% | 3.252227% |
| Common-local terminal phi L2 | 3.621522% | 2.915862% |
| Common-local pooled interface-band phi L2 | 13.779838% | 14.242793% |
| Common-local terminal interface-band phi L2 | 14.863179% | 11.636168% |
| Native nonlocal pooled phi L2 | 3.585413% | 3.564209% |
| Native nonlocal terminal phi L2 | 4.089694% | 3.234165% |
| Common-local pooled q L2 | 60.013480% | 58.061064% |
| Trainer wall, including JIT/evaluation/refinement | 212.38155 s | 217.82857 s |
| Mean compiled SOAP update after100 | 16.9097 ms | 17.3754 ms |

Common-local interface band is \(|\phi_{\mathrm{ref,local}}|<0.9\).
SOAP5000 pooled common-local phase errors are3.882841%/3.888258%.
Pooled means the discrete L2 across all21saved times and64x64points, not an
arithmetic mean of per-time errors. Native and common targets are different.
Field plots below use native reference; the comparison curves use common-local.

Interpretation: no material pooled phase degradation when removing the map in
this pair (direct is0.0516percentage points lower). Cubic favors terminal phase
and terminal band; direct terminal phase is0.7057points/about24.2% higher.
The time-resolved curve shows cubic worsens early times and improves late
times. Thus the map is optional and does not show a uniform or stable advantage.
Known same-seed variability and one fresh seed prevent a statistical conclusion.
Do not attribute the entire useful strategy to this map; other constraints,
flux treatment and training stages remain in both arms. Large q errors remain,
so these phase metrics do not imply all fields are accurate to3%.

Independent verification: config differences are only phase-map presence and
run name; initial raw parameter checkpoint bytes and all36frozen source files
match. Same seed, code and sample counts are checked; full GPU sample traces
were not stored. All new loss logs have exactly two parts, no soft mass loss.
CPU final-checkpoint replay uses the pre-existing FP64 absolute tolerance1e-10:
maximum phi discrepancy7.11e-15(additive)/1.80e-14(cubic), q4.91e-15/5.84e-15.
CPU mass drift is6.17e-16/2.88e-15. Both pass saved history/field audits.
The map changes physical future fields at initialization even with identical raw
parameters; exact prescribed IC at t=0 is the same.

Reproduce analysis from repository root:

    JAX_PLATFORMS=cpu .venv/bin/python docs/tasks/archive/twophase-map-ablation-analyze.py

The existing train.py entry uses configs/train_vortex_consistent.py for direct
output and configs/train_vortex_conserved.py for cubic. Both used seed42,
empty XLA_FLAGS and JAX_ENABLE_COMPILATION_CACHE=false in separate serial
processes. Run source/config snapshots preserve the exact launch configuration.
No production Python, config, reference, default, dependency or Git history
change was made in this task. SOAP issue remains deferred.

Artifacts:

- [Analysis and CPU replays](twophase-map-ablation-analysis.json)
- [Analysis script](twophase-map-ablation-analyze.py)
- [Convergence/common-target per-time errors](twophase-map-ablation-convergence.png)
- [Direct phase/native-reference fields](twophase-map-ablation-additive-phi.png)
- [Cubic/native-reference fields](twophase-map-ablation-cubic-phi.png)
- Per-arm config, summary, refinement_summary, metrics and loss JSON/JSONL files
  with the same prefix alongside this record; checkpoints remain at the run
  paths above. No hash checks or copied external checkpoint paths.

Documentation/registry changes: case README qualifies the older favorable
control and records the fresh pair; artifacts/index.yaml adds both run paths
and direct comparison metadata. Task-end memory: DECISIONS.md, PROGRESS.md and
this completed record. Artifact inventory passes (7cases/353registered
directories), all80local links in the four affected documents resolve, archive
whitespace and git diff --check pass. No GPU compute process remains. No
unrelated numerical suite was rerun for this experiment/documentation-only
change. Both cold starts and the independent CPU audit completed successfully.
