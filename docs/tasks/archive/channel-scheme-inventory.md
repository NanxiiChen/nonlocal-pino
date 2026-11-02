# Current channel NS–CH scheme inventory

## Goal and acceptance

Explain the just-completed stronger-constriction operator's actual architecture,
optimizer, sampling, constraints, differentiation and residual normalization.
Use its saved config and executed code; distinguish optional experiments and
repository defaults. No new training, numerical changes or Git operations.

## Context

- Completed parent: [stronger-constriction trial](channel-stronger-constriction.md).
- Exact saved config: /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-stronger-online-width128-01/config.json.
- Sources: examples/channel_nsch2d/{network,losses,problem,operators,time_batching,train}.py;
  src/nonlocal_pinn/{nn/point_conditional,training/optimizers}.py.

## Status

Completed read-only audit; no blocker or required resume work. No training,
numerical/configuration change or Git operation. Explain the verified inventory
to the user; retain the preceding stronger-trial scope and limitations.

## Verified inventory

- Full NS–CH, T6; geometry only varies: independent upper/lower indentations
  uniform[.06,.20], nominal(.18,.12). Fixed compact smoothstep initial droplet
  center(.8,.5), radius.22, transition.05; carrier+1, droplet-1.
- Shared heterogeneous point operator, raw phi/mu/psi/p four channels. Boundary
  tokens(x,y,nx,ny,type), shape(192,5); domain tokens(x,y,phi0), (15633,3).
  Separate MLP lifts, two heterogeneous normalized integral cross-attention
  blocks, equally averaged boundary/domain messages, width128/heads4/head32,
  FFN256, LayerNorm eps1e-5. No query self-attention. Fourier directions64,
  Gaussian scale2, fixed; 128 sine/cosine features plus coordinates.
- Shared spatial preparation and 3-layer width128 GELU decoder with temporal
  FiLM. Linear/log time coordinates with logscale.25, four frequencies1..2
  each, 17 temporal inputs. One final four-channel linear layer.
- Hard phi IC/inlet, inletmu0; streamfunction lifting imposes initialvelocity0,
  no-slip walls, inlet parabolic profile/flux. ADcurl velocity enforces physical
  incompressibility. Neumann phi/mu wall/outlet and outlet hydrotraction are soft.
  Outlet zero traction uses exterior reference pressure0, no point pressure BC.
- Spatial PDE/BC first derivatives and Laplacian: GMLS degree3/36neighbors,
  exponent16, QR hostFP64 construction, conditionlimit1e9; training arraysFP32.
  Streamfunction-to-velocity curl remains spatial JVP AD. Physical time rates
  use JVP, including mixed derivatives. CH flux uses Dx(phi*ux)+Dy(phi*uy).
  sbp4_mapped supplies loss quadrature, not an exact GMLS mass identity.
- Nine scalar MSE terms, residual divisors: phase.2, chemical1, mx1,my1,
  wallphi1,wallmu4,outlettraction1,outletphi1,outletmu4. Sum after spatial
  quadrature/temporal/condition means. Chemical output amplitude remains4;
  stream/pressure output scales.2/1. No soft continuity, data, IC, velocity-BC
  or mass loss. No tanh/inverse-Jacobian transform, eliminatedmu, causal,
  stagger, adaptive loss balancing, regional weight or time window enabled.
- Per geometry full mapped193x81=15633 spatial nodes,15089 interior residual
  nodes; no core+halo. Forty fresh continuous stratified shared times/update,
  eight-time gradient blocks x5 at identical parameters, one optimizer update.
  Different geometries share computational layout, not physical point arrays.
- Rolling geometry capacity16, condition batch2, replace2 every50updates,
  workers2/prefetch2 batches. Construct geometry/operators only, no PDE labels.
- SOAP upstream soap_jax, beta1=beta2=.9, preconditionevery5/maxdim256/no1D,
  global gradient clip1. LR max(1e-5,1e-3*.7**(step/2000)), continuous Optax
  exponential schedule; 5000 cold steps. FP32/highest matmul/ordinary compile.
  Runtime seed is provenance, not a tuned strategy setting.
- Independent refined local FEM is evaluation only,13 snapshots. Reported
  phi1.9470%,u3.0587%,p3.9455% concerns one nominal development geometry,
  not the reserved16 input-only geometries. Full trial audits are in parent.

## Session verification

Inspected exact saved JSON and live source for architecture, payloads, lifting,
residuals, time accumulation and optimizer. Confirmed installed Optax defaults
staircase=False. Documentation-only changes; no code tests/compilation needed.
Final diff-whitespace and new-record link checks pass.
