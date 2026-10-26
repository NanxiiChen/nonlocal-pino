# Five-second SOAP versus Adam comparison

Goal: change current melting defaults to physical5 s and directly compare SOAP
and Adam using the IC-only latent formulation and identical non-optimizer settings.

Acceptance: base/cold-start defaults and README use Tc5 s with normalized
endpoint1; regression checks cover matching5 s reference and intentional4 s
cross-scale loading; fresh SOAP/Adam runs reach5 s with matched architecture,
seed, graph, spatial/time samples, lr/schedule and5000 updates. Report terminal
full-grid phase L2 against both references, trajectories and measured runtime.
Retain direct-path loading/no hash processing and historical4 s evidence.

Protocol: production train.py, seed42, K_core2000, Nt64, lr0.001 decaying0.95 per
1000 updates, same128^3 graph and IC-only latent residual. Optimizer selection is
the only training-setting difference. Reference:
reports/icemelting-weak-prior-20261024/reference.npz, verified5 s graph snapshots.
This evaluates the configured optimizer choices, not independently tuned optimum
hyperparameters for each algorithm or seed-averaged general performance.

User steering: initial learning rate must be1e-3 for defaults and both runs;
this arrived before training was launched. No0.005 comparison runs were started.

Budget steering: the user subsequently requests5000 updates instead of20000.
Default training iterations are now5000. SOAP had passed5000 at that point;
its process was interrupted and ckpt_005000.eqx is the comparison endpoint.
Adam starts fresh for5000 only. Later SOAP iterates are excluded from comparison.

Status: completed at the user-revised5000-update budget. SOAP uses its saved
5000-step checkpoint after the requested interruption; Adam completed5000 fresh
updates. Both checkpoints independently reloaded and evaluated. Blocker: none.
Next: use the new defaults or explicitly select Adam; no training remains running.

Runs: /root/autodl-tmp/tf-logs/icemelting3d/20261024-5s-soap-lr1e3-seed42 and
/root/autodl-tmp/tf-logs/icemelting3d/20261024-5s-adam-lr1e3-seed42.
Live launch record and stdout:
reports/icemelting-soap-adam-5s-20261024/launch.json, soap.log and adam.log.
Initial driver: /tmp/run_icemelting_optimizer_comparison.py (stopped after SOAP
interruption); revised Adam driver: /tmp/run_icemelting_adam_5k.py (completed).
Both jobs use the current base configuration plus optimizer, reference path and
run name flags. No0.005 jobs ran. Reference identity matches the5 s default
exactly; initial full-grid terminal graph L2=0.7718035011.

Verification so far: modified reference-monitor test passes, covering new5 s
snapshot indices and explicit historical4 s scaling. Base/cold-start configs
both have Tc5/lr0.001; N1=5e-5, N2=0.0987654321, N3=7.8567420132. README updated.
Installed Adam uses beta1=.9/beta2=.999/eps1e-8; SOAP uses beta1=.9/beta2=.99
and its configured preconditioning. These are the existing factory choices;
the comparison is not a matched-moment ablation of preconditioning alone.

## Matched5000-update results

Both rows use physical5 s and all128^3 nodes at the terminal time. The selected
weights are ckpt_005000.eqx in each run directory. Config/weight/reference loading
uses direct file paths; no checksums or hash manifests are computed or required.

| Metric | SOAP | Adam |
| --- | ---: | ---: |
| Full-grid terminal L2 vs same-graph numerical reference | 6.109701081e-5 | 7.395354226e-4 |
| Full-grid terminal L2 vs approximate analytic profile | 4.759003093e-4 | 3.411065686e-4 |
| Last1000 updates, sampled terminal graph L2 median | 4.917443637e-5 | 2.455555990e-4 |
| Last1000 updates, sampled terminal graph L2 range | 1.781e-5–7.159e-5 | 9.700e-5–9.585e-4 |
| Max full-grid phase error vs graph reference | 0.00215648 | 0.0123180 |
| Interface MAE, reference abs(phi)<0.9 | 0.000460410 | 0.00584653 |
| Largest six-direction radius relative error | 0.0326077% | 0.346874% |
| Time from initial evaluation to5000 checkpoint (JIT/evaluation included) | 227.06 s | 200.49 s |
| Median steady update, evaluation boundaries excluded | 26.66 ms | 28.10 ms |

SOAP's selected terminal graph error is12.10x lower; the tail sampled median is
4.99x lower. The tail statistic avoids treating a fluctuating final iterate as
a universal ratio. Adam initially reduces error faster, then oscillates more;
SOAP gives lower/stabler late graph-reference errors in this shared-lr seed42
experiment. Speeds are close once compiled. Total measured training-segment
time includes compilation and intermediate evaluation; it is not a controlled
hardware microbenchmark, and SOAP computation beyond5000 is excluded.

Analytic ranking reverses: Adam is slightly closer to that approximate profile.
The graph reference itself has terminal analytic L2=4.24757e-4. Writing
prediction-minus-analytic = prediction-minus-reference + reference-minus-analytic,
the error-vector cosine between those two terms is approximately-0.9725 for
Adam and+0.8156 for SOAP (norm decomposition on the reference grid). Thus Adam's
PDE-solution error partly cancels the analytic profile's discrepancy. Analytic
error alone would mis-rank accuracy against the fixed graph equation here.

This is a comparison of configured optimizers at one common learning rate and
seed, not proof of a universal factor or of the best independently tuned Adam.
The default remains SOAP; no motion prior, changed PDE or reference training
labels enter either run. Older4 s sota tagging is not re-ranked across horizons.

## Independent verification and evidence

[JSON](icemelting-soap-adam-5s.json) contains configurations, selected checkpoint
paths, trimmed histories, timing samples, initialization checks, full-grid error
re-evaluation, geometry and analytic-error decomposition. The
[figure](icemelting-soap-adam-5s.png) uses sampled terminal L2 curves plus full-grid
markers. The [analysis script](icemelting-soap-adam-5s.py) reproduces these
calculations from the retained local runs when launched at the repository root.

Fresh initial models are exactly equal as Equinox trees and match their seeded
builders. Config differences are optimizer, run name, and SOAP's original saved
20000 limit versus Adam's5000. Only equal5000-step states are compared. Sampling
and learning-rate schedule depend on the update index, not the planned limit;
changing the stop budget does not change SOAP's first5000-step trajectory.
Independent full-grid re-evaluation gives6.109696587e-5 and7.395353996e-4,
matching logged graph errors within floating-point roundoff.

## Task end

Changed train_base defaults to Tc5 s, lr_init0.001 and5000 updates; inherited
cold-start settings use5 s/lr0.001 (2000-update quick preset remains explicit).
README parameters, optimizer selection and interpretation are synchronized.
Updated the focused reference-monitor test for default5 s and explicit historical
4 s compatibility. Added the paired analysis JSON/figure/script and artifact-index
entries for both5 s runs. No old weights, source snapshots or saved configs were
rewritten, including the interrupted SOAP config. No file hashes were added.

Verification: focused reference-monitor test passed; base/cold-start config and
reference identities checked; actual SOAP/Adam GPU execution and checkpoint
reload/initial-state/full-grid/radius checks passed; plot inspected; docs links,
math delimiters and git diff --check passed. No Git history operations performed.
Memory files updated: PROGRESS.md, DECISIONS.md and this archived task.
