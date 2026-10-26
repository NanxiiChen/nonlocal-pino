# Three-case SOTA training defaults — 2026-10-25

## Goal and outcome

Completed. User requested training entries in each train.py, SOTA default
options and README records for corrosion2d/electropolish2d/icemelting3d.
Canonical train_base configs now select the exact verified terminal-field
winners, including seed. KKS ranking uses final5000 phi+c relative L2;
melting retains its selected5 s graph-reference phi result. Best observed
field accuracy is explicitly distinguished from cross-seed/transport accuracy.
No full training, physical-model change, external artifact rename or Git
history operation occurred. Blocker none; no pending action.

## Selected defaults

| Case | Horizon / seed | Formulation and data | Terminal phi / c L2 |
|---|---|---|---|
| corrosion2d | 50 s /42 | bounded eta0.1; raw-KKS edge CH/L1; s_bc; nodal IC/BC | 0.0014330496603163906 /0.001718589539743543 |
| electropolish2d | 5 s /7 | original residual; physical data; random IC/BC | 0.0011378188435460965 /0.0013578160145092115 |
| icemelting3d | 5 s /42 | IC-only latent residual | 6.10970108092214e-5 /— |

All use SOAP,initial lr0.001,5000 updates. KKS grids101x51,Kcore256,pool100,
Nt16,delta0.02; melting128^3,Kcore2000,pool200,Nt64,delta0.01. Resolved
configs, reference paths and selected checkpoints are indexed in
[artifacts/index.yaml](../../../artifacts/index.yaml). Metrics compare physical
fields over every graph node at the final physical time, against independent
integration of the same graph. Local-FEM/approximate analytic and pooled-time
errors are not substituted for this criterion.

Corrosion winner is smallest final field score among40 saved completed-run
summaries; electropolish winner among11 such summaries and also lowest phi
among45 current-cohort saved checkpoints. These are documented observed-study
selections, not an exhaustive claim about every historical unsaved model.

## Limitations retained

Corrosion bounded seed7 phi/c0.4220/0.4461 fails; selected seed42 late phi
median~0.0181 and terminal free-node Dq relative error1.50041 remain significant.
Original-residual/raw-edge and rate-tail options remain documented. Electropolish
random seed42 phi/c~0.29093/0.29901 fails; nodal-data s_bc is more consistent
across the two observed seeds but has worse seed7 endpoint. Selection includes
the fixed winning seed, not a universally superior residual/data strategy.
Melting's linear-radius analytic profile remains approximate; its historical
4 s directory tag is preserved separately from current5 s SOTA provenance.

## Implementation and diff-level changes

- New corrosion configs/train_base.py promotes bounded/raw/s_bc/nodal seed42.
- Electropolish configs/train_base.py promotes original/physical/random seed7;
  exact historical large-grid values preserved in configs/train_legacy.py.
- residual_pilot imports preserved legacy bases, avoiding config recursion or
  silent changes to archived research CLI defaults.
- Both canonical train.py entries detect explicit matched configs and delegate
  to training_entry.train_matched; legacy configs retain the original loops.
- Four analytic data samplers move unchanged to training_data.py and remain
  exported under the old helper names. The shared engine no longer imports
  train.py, avoiding duplicate absl flag registration for direct-script use.
- Canonical adapter validates case and rejects unsupported legacy switches,
  enables the audited deterministic GPU flag while preserving unrelated XLA
  settings, and fails on conflicting settings. No seed or loss fallback.
- Shared engine gives empty run_name a fresh UTC microsecond name; existing
  explicit directories still fail. Saved config records the actual new name.
  Snapshot inventory includes canonical entry/config, adapter and samplers.
- Root and three example READMEs document command/config/metrics/provenance
  and limitations; DEVELOPMENT explains dispatch. Migration notes distinguish
  this explicit new default authorization from historical split preservation.
- Artifact default_training/SOTA pointers updated; historical4 s melting SOTA
  identity/metric retained, without moving or rewriting any trained artifacts.

The new public commands, from this repository's activated .venv, are:

```bash
python -m examples.corrosion2d.train
python -m examples.electropolish2d.train
python -m examples.icemelting3d.train
```

Equivalent direct-script entries work. All load configs/train_base.py and
start cold, with fresh names. Explicit legacy presets remain valid; matched
configs require a compatible sparse reference. PDE/physical params/KKS,
pre-scales/optimizer update math remain those of the selected experiments.

## Verification and artifacts

- [Config/initialization/sampler audit source](phasefield-sota-defaults-check.py)
  and [complete output](phasefield-sota-defaults-check.json).
  All saved config fields match, except new run names; corrosion adds disabled
  lr-tail/CH-norm-continuation fields, melting uses requested5000 instead of
  originally planned20000 and optional graph-monitor path remains disabled.
- Same-CPU saved-config/default initial parameters match exactly for all3
  cases. CPU versus saved GPU initial normal draws differ at most4.76837e-7;
  this is separately measured, not an exact cross-device replay claim.
- Four extracted samplers produce exactly equal arrays to the original CLI
  functions. Matching KKS reference load/shape/grid/BC checks pass.
- 30 existing numerical checks pass:20 KKS loss/map/graph/GradNorm/transport,
  5 melting constraint/residual/monitor/update,2 SOAP tail and3 time-sampling.
  Initial combined command's new CLI stub tests returned a MagicMock exit
  status; fixed the stub to return None like the real trainer and reran only
  the affected entry suite. All3 entry tests pass, including real absl direct
  and module dispatch/overrides for both cases and unsupported-option rejection.
- Three actual reduced-network CPU smoke jobs complete:2 corrosion updates,
  2 electropolish updates and1 corrosion update with empty run_name. Finite
  losses/gradients, joint activation, correct IC/BC counts, source snapshots,
  model/step/final saves, JSONL and actual fresh name serialization checked.
  These are entry/update checks, not new SOTA accuracy measurements. Temporary
  paths and scope in [smoke record](phasefield-sota-defaults-smoke.json).
- Full-model initial builders, selected checkpoint/reference paths, syntax,
  JSON, local Markdown links, artifact inventory(5 cases/295 directories),
  legacy function/config preservation and git diff --check pass. No expensive
  unchanged suite repeated after its successful numerical checks. All jobs exit.

## Task end

Complete; no blocker or pending work. Changed memory: docs/PROGRESS.md,
docs/DECISIONS.md and this archived task. Prior results/checkpoints stay intact.
