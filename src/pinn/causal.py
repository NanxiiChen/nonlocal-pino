"""Causal temporal weighting for PINN training.

Splits the time domain into chunks, computes per-chunk PDE loss, and
assigns exponentially-decaying weights:
    w_k = exp(-eps × Σ_{i<k} L_i),   w_0 = 1

This forces the network to learn early times before later times.
AC and CH each track their own eps; the slower-converging equation
determines the pace via min(eps_ac, eps_ch).
"""

import jax.numpy as jnp
import jax


class CausalWeightor:
    def __init__(self, n_chunks, t_range=(0.0, 1.0)):
        self.n_chunks = n_chunks
        self.bins = jnp.linspace(t_range[0], t_range[1], n_chunks + 1)

    # ---- per-chunk loss ---------------------------------------------------

    def chunked_loss(self, r, t_batch):
        """r: (N_x, N_t) residual.  Returns per-chunk MAE of shape (n_chunks,)."""
        indices = jnp.digitize(t_batch, self.bins) - 1
        r1_t = jnp.mean(jnp.abs(r), axis=0)                      # (N_t,)
        sum_abs = jax.ops.segment_sum(r1_t, indices, num_segments=self.n_chunks)
        count = jax.ops.segment_sum(jnp.ones_like(r1_t), indices,
                                     num_segments=self.n_chunks)
        return sum_abs / (count + 1e-12)

    # ---- weights ----------------------------------------------------------

    def compute_weights(self, loss_chunks, eps):
        """w_k = exp(-eps × Σ_{i<k} L_i), w_0 = 1."""
        cum = jnp.cumsum(loss_chunks[:-1])
        w = jnp.concatenate([jnp.array([1.0]), jnp.exp(-eps * cum)])
        return jax.lax.stop_gradient(w)

    # ---- weighted loss ----------------------------------------------------

    def weighted_loss(self, r, t_batch, eps):
        """Returns (weighted_MAE, chunk_weights, loss_chunks)."""
        chunks = self.chunked_loss(r, t_batch)
        w = self.compute_weights(chunks, eps)
        indices = jnp.digitize(t_batch, self.bins) - 1
        w_per_point = w[indices]
        return jnp.mean(jnp.abs(r) * w_per_point[None, :]), w, chunks

    # ---- eps update (pure — no mutation) ----------------------------------

    def update_eps(self, w, eps, step_size, max_eps, max_min_weight,
                   min_mean_weight):
        """Return updated eps given causal weight vector.

        - If last chunk weight > max_min_weight → later times well-learned,
          increase eps (make weighting more aggressive).
        - If mean weight < min_mean_weight → time evolution not learned yet,
          decrease eps (relax weighting).
        """
        w_last = w[-1]
        w_mean = jnp.mean(w)
        eps = jnp.where((w_last > max_min_weight) & (eps < max_eps),
                        eps * step_size, eps)
        eps = jnp.where(w_mean < min_mean_weight,
                        eps / step_size, eps)
        return jnp.clip(eps, 0.0, max_eps)
