# Global mixed-field controls

All main rows are final physics-only checkpoints, seed42, same frozen T1.5 physical/reference case, SOAP5000+LBFGS500. No reference loss. Shared/separate have identical40 source files; separate/block initial checkpoint files are identical. Field layout changes parameter count. The block control also changes differential to integrated residual and norm, so benefits cannot be attributed to output scaling alone.

| Model | Parameters | Phase pooled FEM % | Phase terminal FEM % | Mu pooled native % | Mu terminal native % | Trainer s | Sampled energy max rise |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eliminated R/t | 16344 | 21.5396 | 43.1779 | 258.0723 | 2716.0436 | 197.77 | 0.0670517 |
| Eliminated strong | 16344 | 57.8409 | 65.5523 | 90.3234 | 319.7936 | 251.90 | 0.000171316 |
| Mixed shared | 17904 | 58.4902 | 63.5036 | 70.2777 | 301.4807 | 265.58 | 0.000656468 |
| Mixed separate | 32688 | 71.2418 | 71.3012 | 74.0590 | 360.5670 | 285.05 | 0.00465224 |
| Mixed block inverse | 32688 | 46.1064 | 38.6016 | 50.5093 | 130.0648 | 386.24 | 0 |

Block control improves terminal phase/mu and removes sampled energy rises, but pooled phase is worse than prior21.54% model. None is a satisfactory high-accuracy global solution; no default/SOTA promotion or seed stability claim.

## Explicitly supervised-only capacity diagnostic

Same separate global network and identical initial weights,5000 SOAP with native field targets: phase pooled0.43846%, terminal0.42996%; mu pooled2.7160%, terminal5.16961%. These are not PINN accuracy results, and no weights enter any physics run. This establishes field expressivity, not accurate derivatives or a joint small-PDE-residual approximation. See chmixed-representation-audit.json for actual physical residuals.

## Validation-only temporal coefficients

Free rank24 nodal coefficients: phase0.07697%, mu0.94542%, strong evolution loss1.00657. Known initial derivative/mu constraints retain phase0.07372%/mu0.73586% and reduce that loss to0.054208. This is a constrained field projection, not a residual-minimizing neural/PINN fit or a proof of basis inadequacy. Native reference replay1.90e-14.

CPU final/best replays and independent chemical/mass/energy audits are in chmixed-analysis.json. Both new numerical tests pass. Paths load directly; no hashes, new reference solver or Git history changes.

The supervised-only weights have actual block PDE loss8.98496e-5 versus the
physics-trained.00276241 (30.745x). Their strong loss.459483 remains higher,
so this is evidence of a feasible better joint field/block approximation, not
proof that any accurate field automatically has an accurate time derivative.
Source/initial/config parity: chmixed-provenance.json. Boundary audit:21.10%
area carries57.39% terminal native squared error for block; bulk error28.47%.
