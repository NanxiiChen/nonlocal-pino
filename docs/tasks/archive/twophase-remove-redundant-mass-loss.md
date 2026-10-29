# Remove mass loss under the exact mass constraint

Goal: remove the redundant soft mass penalty when the two-phase physical field
already enforces exact_grid_mean; preserve the unprojected legacy soft-loss path.

Acceptance: hard-mass loss returns only two PDE components and does not require
or evaluate mass_weight/mean penalty. Unconstrained configs retain three parts
and the explicit mass weight. Update canonical hard preset, validation/docs and
focused checks; preserve checkpoint parameter leaves and external artifacts.

Context: user agrees with hard mass conservation and asks whether a redundant
loss remains. It does in twophase2d; precipitate2d already has only two terms.
No full retraining or Git history operation requested. SOAP stays deferred.

Status: complete (2026-10-27).
Blocker: none.
Next action: none. No full retraining or L2 improvement claim.

Changes: losses.py branches once on explicit phase_mass. The hard path reads
only the two PDE weights and returns two residual MSE components; it neither
reads mass_weight nor computes the mean penalty. The unprojected legacy path
still requires the mass weight, evaluates its penalty and returns three parts.
The hard preset train_vortex_consistent removes the inherited obsolete field;
all hard descendants inherit this. problem.py validates the appropriate schema.
Old hard serialized configs with an extra mass_weight remain accepted/ignored.
No model leaves, physics, references or external files were changed.

README and DEVELOPMENT explain two/three component logs and the distinction
between a mass diagnostic and a loss. Old saved logs remain untouched.
Existing tests now verify two hard components/weighted objective, independence
from the unused legacy target argument, acceptance of obsolete saved weight,
phase projection/time derivative/parameter gradients and a nonzero soft mass
penalty under a deliberately shifted target. Hard local AD and soft/nonlocal
paths are exercised by the narrow focused command.

Validation: three focused unittest checks pass (32.936s CPU); all18 twophase
presets validate (6 hard without mass weight,12 soft with explicit weight).
git diff --check and documentation links pass. No expensive unrelated suite,
GPU training, dependency or Git history operation. A mathematically redundant
penalty removal does not imply bitwise-identical future SOAP trajectories.

Focused output: twophase-remove-redundant-mass-loss-tests.txt alongside this
archive. Changed memory: PROGRESS.md, DECISIONS.md, DEVELOPMENT.md and this task.
