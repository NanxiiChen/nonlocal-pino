"""Causal temporal weighting for PINN training.

Two modes:
  - Per-t (n_chunks=None, default): each time point is its own segment.
    Finer temporal resolution, works well with random time sampling.
  - Chunked (n_chunks=int): legacy mode, bins time into fixed chunks.

Weights: w_k = exp(-eps * cumsum(L_{<k})),  w_0 = 1.

Fixed t_weights (t_ramp) are applied independently in the loss functions,
not through the causal module.  The two are mutually exclusive.
"""

import jax.numpy as jnp
import jax


class CausalWeightor:
    def __init__(self, n_chunks=None, t_range=(0.0, 1.0)):
        """n_chunks=None: per-t mode. n_chunks=int: legacy chunked mode."""
        self.n_chunks = n_chunks
        self.bins = (
            jnp.linspace(t_range[0], t_range[1], n_chunks + 1)
            if n_chunks is not None else None
        )

    # ---- per-segment loss --------------------------------------------------

    def segment_loss(self, r, t_batch):
        """Returns per-segment MAE.  In per-t mode: sorted by time (N_t,).
        In chunked mode: per-chunk (n_chunks,)."""
        if self.n_chunks is not None:
            indices = jnp.digitize(t_batch, self.bins) - 1
            r1_t = jnp.mean(jnp.abs(r), axis=0)
            s = jax.ops.segment_sum(r1_t, indices, num_segments=self.n_chunks)
            c = jax.ops.segment_sum(jnp.ones_like(r1_t), indices,
                                     num_segments=self.n_chunks)
            return s / (c + 1e-12)
        sort_idx = jnp.argsort(t_batch)
        r_sorted = r[:, sort_idx]
        return jnp.mean(jnp.abs(r_sorted), axis=0)  # (N_t,)

    # ---- weights -----------------------------------------------------------

    def compute_weights(self, loss_segments, eps):
        """w_k = exp(-eps * cumsum(L_{<k})), w_0 = 1."""
        cum = jnp.cumsum(loss_segments[:-1])
        w = jnp.concatenate([jnp.array([1.0]), jnp.exp(-eps * cum)])
        return jax.lax.stop_gradient(w)

    # ---- weighted loss -----------------------------------------------------

    def weighted_loss(self, r, t_batch, eps):
        """Returns (weighted_MAE, segment_weights, loss_segments)."""
        seg_loss = self.segment_loss(r, t_batch)
        w = self.compute_weights(seg_loss, eps)

        if self.n_chunks is not None:
            indices = jnp.digitize(t_batch, self.bins) - 1
            w_per_point = w[indices]  # (N_t,)
        else:
            sort_idx = jnp.argsort(t_batch)
            unsort_idx = jnp.argsort(sort_idx)
            w_per_point = w[unsort_idx]  # (N_t,)

        return jnp.mean(jnp.abs(r) * w_per_point[None, :]), w, seg_loss

    # ---- eps update (pure — no mutation) ----------------------------------

    def update_eps(self, w, eps, step_size, max_eps, max_last_weight,
                   min_mean_weight):
        """Return updated eps given causal weight vector.

        - If last segment weight > max_last_weight: later times well-learned,
          increase eps (more aggressive weighting).
        - If mean weight < min_mean_weight: time evolution not learned yet,
          decrease eps (relax weighting).
        """
        w_last = w[-1]
        w_mean = jnp.mean(w)
        eps = jnp.where((w_last > max_last_weight) & (eps < max_eps),
                        eps * step_size, eps)
        eps = jnp.where(w_mean < min_mean_weight,
                        eps / step_size, eps)
        return jnp.clip(eps, 0.0, max_eps)
