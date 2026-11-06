# Particle corrosion sampling clarification

Date: 2026-11-06 (Asia/Shanghai). Status: complete.

User asks whether current particle-corrosion training geometries roll online.
Acceptance: inspect actual selected-run configuration/source and distinguish
geometry replacement, random selection of existing core pools and time draws.
This is confirmation only; no online implementation or new training requested.

Current source/default and saved runtime identify the fixed64geometry bank
20261106-eight-input-bank64-01. Each geometry has eight fixed core/halo pools.
The runner loads the bank once and randomly selects four geometry IDs and pool
IDs per update. Each update uses eight independent uniform time draws, generated
in the complete reproducible trace before optimization. No bank replacement or
background geometry producer exists in this runner. Selected nonlocal and paired
localAD use identical saved sampling arrays. Existing1.9587%reserved phase result
therefore describes finite-bank training, not online-infinite geometry training.

Saved run source independently confirms the same sampling: train.py65loads once,
125copies the bank once,89--91generate the complete draw trace,128--130gather
existing arrays,151reads each step. The later CLI default-preset promotion at
line191 is separate from this unchanged sampling logic. Actual manifest has
64geometries,8pools,256cores and paddedhalo2560; each step's8times are shared
across its4geometry draws. No background producer or bank replacement exists.

Task-end: source/runtime/manifest audit complete; clarify finite-bank versus
rolling training to the user. Archived record and PROGRESS/DECISIONS updated.
No blocker or next job. No source/model/default changes, GPUjobs, Git operations
or tests required for this documentation-only confirmation. Diff check passes.
