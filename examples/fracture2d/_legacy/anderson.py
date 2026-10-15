"""Type-II Anderson with explicit numerical safeguards, no field projection.

Walker/Ni (2011), DOI 10.1137/10078356X, Eq. (1.2), Eq. (3.1) and Section 4.
The normalized-infinity merit, backtracking toward the Picard point and
physical callbacks are project numerical policy; they do not change the map.
"""

import numpy as np
from scipy.linalg import qr, solve_triangular


def _vector(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.float64 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite FP64 of shape {shape}")
    return value


def map_merit(x, image):
    error = float(np.max(np.abs(image - x)))
    scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(image))))
    return error / scale if scale > 0 else 0.


def propose(history_x, history_f, x, image, *, depth, beta, rank_strategy="restart"):
    x = np.asarray(x)
    if x.ndim != 1 or not len(x):
        raise ValueError("Anderson state must be a nonempty vector")
    x, image = (_vector(value, x.shape, name) for value, name in ((x, "state"), (image, "image")))
    if isinstance(depth, bool) or int(depth) != depth or depth < 1 or not np.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError("Anderson depth must be positive integer and beta in (0, 1]")
    if len(history_x) != len(history_f):
        raise ValueError("Anderson state/residual histories must have equal lengths")
    if rank_strategy not in ("restart", "drop_oldest"):
        raise ValueError("Anderson rank strategy must be restart or drop_oldest")
    for past_x, past_f in zip(history_x, history_f):
        _vector(past_x, x.shape, "history state")
        _vector(past_f, x.shape, "history residual")
    m = min(depth, len(history_x))
    info = {"history_size": m, "qr_rank": 0, "coefficient_norm": None, "proposal_reason": "insufficient_history",
            "discarded_oldest": 0, "rank_strategy": rank_strategy}
    if not m:
        return None, info
    xs, fs = [*history_x[-m:], x], [*history_f[-m:], image - x]
    dx = np.column_stack([xs[i + 1] - xs[i] for i in range(m)])
    df = np.column_stack([fs[i + 1] - fs[i] for i in range(m)])
    while True:
        threshold = max(df.shape) * np.finfo(np.float64).eps * max(float(np.max(abs(df))), np.finfo(np.float64).tiny)
        Q, R, pivots = qr(df, mode="economic", pivoting=True, check_finite=True)
        rank = int(np.count_nonzero(np.abs(np.diag(R)) > threshold))
        info.update({"qr_rank": rank, "history_size": m})
        if rank == m:
            break
        if rank_strategy == "restart" or m == 1:
            info["proposal_reason"] = "rank_deficient"
            return None, info
        dx, df = dx[:, 1:], df[:, 1:]
        m -= 1
        info["discarded_oldest"] += 1
    gamma = np.empty(m, dtype=np.float64)
    gamma[pivots] = solve_triangular(R, Q.T @ (image - x), lower=False, check_finite=True)
    candidate = x + beta * (image - x) - (dx + beta * df) @ gamma
    _vector(candidate, x.shape, "Anderson candidate")
    info.update({"coefficient_norm": float(np.linalg.norm(gamma)), "proposal_reason": "formed"})
    return candidate, info


def safeguarded_step(x, image, history_x, history_f, *, evaluate, bounds_pass,
                     physics_pass, depth, beta, sufficient_decrease, max_backtracks,
                     restart_policy, rank_strategy="restart"):
    """Return selected state/image/payload and restartable secant history.

    evaluate(state) returns (image, payload); failed linear solves and I/O are
    deliberately not swallowed. Physically rejected extrapolations are audited
    and backtracked. A failed Picard physical check is returned for caller rollback.
    """
    if (not np.isfinite(sufficient_decrease) or not 0 < sufficient_decrease < 1
            or isinstance(max_backtracks, bool) or int(max_backtracks) != max_backtracks or max_backtracks < 0):
        raise ValueError("Anderson decrease must be in (0, 1), backtrack count a nonnegative integer")
    if restart_policy not in ("reject_or_rank", "rank_only"):
        raise ValueError("Anderson restart policy must be explicit")
    candidate, info = propose(history_x, history_f, x, image, depth=depth, beta=beta, rank_strategy=rank_strategy)
    picard = x + beta * (image - x)
    current_merit = map_merit(x, image)
    info.update({"current_merit": current_merit, "merit_factor": 1 - sufficient_decrease * beta, "trials": []})
    selected = None
    if candidate is not None:
        for backtrack in range(max_backtracks + 1):
            alpha = 2.**(-backtrack)
            trial = picard + alpha * (candidate - picard)
            audit = {"alpha": alpha, "bounds_pass": bool(bounds_pass(trial))}
            if not audit["bounds_pass"]:
                audit["reason"] = "bounds"
            else:
                trial_image, payload = evaluate(trial)
                _vector(trial_image, x.shape, "candidate map image")
                merit = map_merit(trial, trial_image)
                physical = bool(physics_pass(payload))
                audit.update({"merit": merit, "physics_pass": physical})
                if physical and merit <= info["merit_factor"] * current_merit:
                    audit["reason"] = "selected"
                    selected = (trial, trial_image, payload)
                else:
                    audit["reason"] = "physics" if not physical else "insufficient_decrease"
            info["trials"].append(audit)
            if selected is not None:
                break
    restart = (info["proposal_reason"] == "rank_deficient"
               or restart_policy == "reject_or_rank" and candidate is not None and selected is None)
    info.update({"selected_kind": "anderson" if selected is not None else "picard",
                 "restart": bool(restart), "restart_policy": restart_policy})
    if selected is None:
        if not bounds_pass(picard):
            raise ValueError("ordinary Picard candidate violates raw bounds; no clipping or fallback field")
        picard_image, payload = evaluate(picard)
        _vector(picard_image, x.shape, "Picard map image")
        selected = (picard, picard_image, payload)
    info["selected_physics_pass"] = bool(physics_pass(selected[2]))
    info["selected_merit"] = map_merit(selected[0], selected[1])
    hx, hf = ([], []) if restart else (list(history_x), list(history_f))
    if not restart and info["discarded_oldest"]:
        hx, hf = hx[-info["history_size"]:], hf[-info["history_size"]:]
    hx = [*hx, x.copy()][-depth:]
    hf = [*hf, (image - x).copy()][-depth:]
    return (*selected, info, hx, hf)
