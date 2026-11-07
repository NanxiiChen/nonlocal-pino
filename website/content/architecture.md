The five operators share a coordinate-query decoder pattern. The condition encoder depends on whether the input is a field on a fixed grid or a domain with sample-dependent geometry.

## Encoding fixed domains

Melting conditions on a \(32^3\) initial phase volume. A nonoverlapping \(4^3\) patch yields one token, giving \(8^3=512\) tokens. Electropolish uses \(4\times8\) patches of a \(32\times64\) image, giving \(8\times8=64\) tokens. Project flattened patches into the model width, retain patch positions, and mix tokens with self-attention. Both selected models use two condition blocks and four heads.

<figure>
  <a href="assets/architecture.svg"><img src="assets/architecture.svg" width="960" height="440" alt="Two condition encoders and their common decoder pattern"></a>
  <figcaption>The decoder receives condition context through spatial cross-attention. Temporal FiLM then modulates those prepared features for each query time.</figcaption>
</figure>

## Encoding irregular domains

The spinodal, channel and particle cases each use two streams. The boundary stream contains coordinates, normals and boundary type. The domain stream contains coordinates and a prescribed initial field. Boundary arc lengths and domain areas serve as quadrature measures. Invalid or padded entries carry zero measure and are masked in the fixed-shape payloads.

Independent lifts put the streams into a common feature width. Heterogeneous cross-attention mixes the streams without requiring patch alignment or a fixed physical point ordering. As a schematic quadrature view of a normalized attention head, a query \(\mathbf q\) reads weighted values \(\mathbf v_j\) using a feature kernel \(\kappa\).

$$
\mathcal A(\mathbf q)=
\frac{\displaystyle\sum_j m_j\,\kappa(\mathbf q,\mathbf k_j)\,\mathbf v_j}
     {\displaystyle\sum_j m_j\,\kappa(\mathbf q,\mathbf k_j)} .
$$

This expression illustrates the role of the physical measures \(m_j\). See the implementation for the per-stream projections, normalization and mixing. Geometry parameters generate meshes, boundary data and known fields. The selected irregular-domain models pass those physical inputs to the encoder.

## Queries and time

Coordinate queries read the encoded condition by cross-attention. Fixed domains can share a physical query set across conditions. Different geometries have their own coordinates, while the same geometry shares its spatial preparation across times. The temporal decoder uses feature-wise modulation (FiLM).

$$
\mathbf z_{\ell+1}
=\sigma\!\left(\boldsymbol\gamma_\ell(t)\odot
       W_\ell\mathbf z_\ell+\boldsymbol\beta_\ell(t)\right).
$$

This is the modulation pattern. The actual decoders also use their configured activations, embeddings and output maps. A time JVP through that map differentiates the resulting physical fields. Known fields can also appear in a hard lift, as with melting's initial signed distance. That use is distinct from query-local conditioning features in the learned attention path.

## Implementation

Read the [point conditional implementation](https://github.com/NanxiiChen/nonlocal-pinn/blob/main/src/nonlocal_pinn/nn/point_conditional.py) for the geometry encoder and see the [melting case](ice-melting.html) and [electropolish case](electropolish.html) for the patch configurations. [Training and validation](training.html) describes how the inputs are assembled and sampled.
