# Electropolish composed-attention CPU/GPU replay difference

Date: 2026-10-31. Status: OPEN, recorded separately from the completed spatial
sampling comparison. No production change or numerical fallback is selected.

## Reproduction and evidence

Run:
`/root/autodl-tmp/tf-logs/electropolish2d/20261031-electro-spatial-01-full`.
Final5000model, same saved configuration/inputs. GPU NVIDIA RTX6000D, JAX0.10.1,
ordinary compilation, FP32/highest matmul precision. CPU and GPU arrays of all
146242model entries, image, coordinates and times compare exactly.

Original failed independent CPU replay:
`/root/autodl-tmp/tf-logs/electropolish2d/20261031-electro-spatial-01-CPU_replay.log`.
Declared absolute field tolerance2e-5 is **not** relaxed. First input has a max
gap0.316676inFP32; promoting the saved parameters toFP64 gives0.316677, so the
chosen replay dtype does not explain this discrepancy. Same-GPU checkpoint
replay on all6monitor/reference inputs passes in each of the3spatial controls.
The64final-error calculations are independently reproduced from saved fields.

Composed network diagnostic, first input:

| Stage | Maximum CPU/GPU absolute gap |
|---|---:|
| Encoded tokens |1.19e-6 |
| Spatial Fourier features |1.91e-6 |
| Spatial linear projection |8.34e-7 |
| Normalized q / kv |1.19e-6 /1.43e-6 |
| Cross-attention output |0.208802 |
| Prepared spatial features |0.406896 |
| Physical output |0.316676 |

An independent NumPy projection/head-split/scaled-softmax/value/output formula
matches CPUattention within1.44e-6 but differs from the composed GPUattention
by0.208802. An isolated attention call given **identical explicit q/kv arrays**
agrees across CPU/GPU within7.16e-7, including projections/logits and softmax.
Thus the issue is reproduced in the composed path, not an identified standalone
attention formula error. The exact compiler/backend cause remains unresolved;
do not attribute it uniquely to a particular library or optimization pass.

Read-only diagnostic controls retained:

- Explicit-model versus captured-model CPU calls agree; CPU32/CPU64 both differ
  from the saved GPU field substantially.
- Disabling cuDNNFMHA leaves the GPU result essentially unchanged.
- Disabling TritonGEMM leaves the GPU result within8.59e-6of its saved field.
- A q/kv optimization barrier does not remove the composed discrepancy.

All arrays and detailed probes remain under
`/root/autodl-tmp/tf-logs/electropolish2d/20261031-electro-spatial-01-comparison`:
`replay-parts-*.npz`, `spatial-parts-*.npz`, `attention-matrices-*.npz`,
`attention-numpy-probe.json`. No original output is deleted or overwritten.

Clean regeneration entry:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m docs.tasks.electro_attention_replay_probe \
  --run=/root/autodl-tmp/tf-logs/electropolish2d/20261031-electro-spatial-01-full \
  --output=/tmp/electro-attention-cpu.npz
```

Repeat in a separate process with JAX_PLATFORMS=cuda and a different output path.
Use the repository .venv. Generated NPZs are external local artifacts, not Git
payloads. The script checks a cross-attention saved model, not the FiLM control.

## Consequences and deferred work

The spatial comparison measures all3final models on the same actual GPU path
against identical independently solved native/localFDreferences. Those empirical
errors/timings and actual-input parity remain measured facts. Restrict their
interpretation to that backend; CPU portability and backend-independent model
equivalence have **not** passed. Shape-dependent compiled execution could also
affect accuracy comparisons, so no default promotion or general sampling-method
superiority follows from this one experiment.

Future work: minimize the composed-producer graph, inspect optimized HLO and
head layouts, check query chunk/set invariance and alternative strict reference
arithmetic. Any production execution change must preserve old saved-config
semantics or be explicit, then repeat matched controls. No hidden correction,
changed checkpoint, loosened threshold or extra training is applied here.
