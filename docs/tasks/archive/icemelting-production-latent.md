# Default latent melting formulation and cold-start monitoring

Location update: the bounded run was subsequently renamed to
`20261024-ic-latent-coldstart-01-sota` at the user's request. Original run IDs,
configuration and source/event contents are preserved. See the
[rename record](icemelting-run-tags.md).

Loading update: current runtime and replay scripts use explicit file paths and
do not compute or require file hashes. The checksum statements below describe
checks performed during the original experiment, not requirements for loading.
See the [direct-path update](icemelting-direct-path-loading.md).

Goal: make the validated IC-only tanh representation and exact transformed
residual the default production melting option; cold-start training and monitor
relative L2 with explicitly labelled references and terminal times.

Acceptance: selectable legacy formulation; exact IC with no prescribed motion;
stable graph/local residual equivalence and focused regression checks; documented
defaults; fresh run with checkpoint/config/source provenance and a measured L2
history including terminal fields. Historical checkpoints remain untouched.

Context: archived weak-prior controls and baseline metric audit. Historical
production Tc=4 s; new pilots Tc=5 s. Analytic is approximate, same-graph numerical
references measure optimisation error, and the two errors must remain separate.

Status: completed. Integration and five focused regression checks passed (46.8 s
CPU); both production cold starts completed2000 updates and final checkpoints
were independently reloaded. Blocker: none. No process remains running.
Next: full20000-update or production5 s validation is follow-up work; this task
establishes default integration and bounded4 s cold-start behaviour only.
The full base budget remains 20000/K_core2000/Nt64/lr0.005. New default selector
is phase.formulation=ic_latent; ic_phase/raw_phase remain explicit alternatives.

Implementation: phase.py wraps the unchanged historical raw builder; losses.py
uses stable edge ratios or the exact local chain rule and omits redundant IC
loss/GradNorm for hard modes. evaluation.py logs pooled/terminal analytic and
optional matching-graph errors, with full-grid terminal checks and physical-time
reference validation. train.py saves sources/config/hashes and cold-start weights,
rejects existing run directories and fails on nonfinite gradients instead of
sanitizing them. Full-grid graph training now passes graph edges explicitly
rather than accidentally selecting local AD; diagnostics use aligned subgraphs.
The FD import typo inherited from the split was corrected to exercise that mode.

Validation so far: tests.test_icemelting_latent (5 tests) passed: exact IC after
parameter change, no built-in speed, raw/new serialization, graph/local chain
rule, saturation gradients, padded sparse and FD SOAP updates in all three modes,
ignored IC targets for hard modes, reference mismatch rejection and physical-time
mapping with padded full-grid inference.

## Cold-start observations

Bounded run: /root/autodl-tmp/tf-logs/icemelting3d/20261024-ic-latent-coldstart-01-sota,
seed42, Tc4 s, 2000 updates, K_core256/Nt16/lr0.001, source snapshot and initial
weights saved. Full-grid terminal graph L2: update0=0.7414720597,
update1000=5.232160312e-5, update2000=3.018079939e-5. Final terminal analytic
L2=0.001028267242. Sampled terminal graph errors fall sharply by600 updates,
then fluctuate; they are not monotone. Summary reports finite completion.

Second run: /root/autodl-tmp/tf-logs/icemelting3d/20261024-ic-latent-default-coldstart-01.
Command: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m
examples.icemelting3d.train --config.train.iterations=2000
--config.eval.graph_reference_path=reports/icemelting-weak-prior-20261024/reference.npz
--config.log.run_name=20261024-ic-latent-default-coldstart-01.
Only budget/reference/run name differ from base; no hyperparameter tuning.
Completed without nonfinite updates; stdout is preserved at
/tmp/icemelting-production-default-20261024.log. Its full-grid terminal graph L2
is 0.7414720597 at update0, 0.0002399012008 at1000, and 0.0003862176282 at2000.
The final analytic full-grid terminal L2 is0.001396905462. Wall time after initial
evaluation is139.47 s (including JIT); bounded-preset equivalent is83.33 s.

## Interpretation and independent checks

Both histories show large early reductions followed by low-error oscillations.
The default-batch run is not monotonic: its sampled terminal graph L2 is
1.889e-4 at800 and3.961e-4 at2000; the later iterate is not necessarily better.
The bounded-preset final error is lower, but its batch sizes and learning rate
both differ, so these runs do not isolate a learning-rate effect. The base
learning rate and batch sizes were deliberately retained, not silently tuned.

Both initial checkpoint hashes are identical and match fresh seeded builders.
Reloading final checkpoints preserves the exact initial field. Source snapshot
hashes and final checkpoint hashes were independently verified. Six near-axis
grid rays at physical4 s have maximum radius relative error0.00611% for the
bounded preset and0.09287% for the default batch. Reference radius is0.1502045834;
ray max phase errors are0.0004254 and0.004419 respectively. These supplement the
global phase L2, which weights the large water volume as well as the interface.

The archived [JSON](icemelting-production-latent.json) contains full configurations,
source hashes, both metric histories, run summaries and independent geometry/IC
checks. The [figure](icemelting-production-latent.png) shows terminal L2 versus
completed updates; curves use8000 fixed grid nodes and circles use all128^3 nodes.
The [report script](icemelting-production-latent.py) regenerates these artifacts
from the preserved local runs when launched from the repository root with the
project .venv. The training config and source snapshots inside each run are the
authoritative record of the executed code.

## Task end

Changed production files: train.py, losses.py, new phase.py/evaluation.py,
configs/train_base.py and new configs/train_cold_start.py, plus example README.
One inherited FD import typo was fixed to keep the advertised baseline usable.
Added tests/test_icemelting_latent.py and archived numerical evidence. Updated
memory files: PROGRESS.md, DECISIONS.md and this archived task. Earlier research
runner and evidence remain separate and historical checkpoints were untouched.

Verification: all five focused numerical tests passed; both GPU production runs
finished2000 updates; fresh initial/final deserialization, source hashes, hard IC
and geometry checks passed; inspected the generated figure; git diff --check
passed. No commit, merge, push, or historical optimizer resume was performed.
