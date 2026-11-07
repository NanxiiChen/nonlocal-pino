The five operators share a coordinate-query decoder pattern. The condition encoder depends on whether the input is a field on a fixed grid or a domain with sample-dependent geometry.

## Encoding fixed domains

Melting conditions on a \(32^3\) initial phase volume. A nonoverlapping \(4^3\) patch yields one token, giving \(8^3=512\) tokens. Electropolish uses \(4\times8\) patches of a \(32\times64\) image, giving \(8\times8=64\) tokens. Project flattened patches into the model width, retain patch positions, and mix tokens with self-attention. Both selected models use two condition blocks and four heads.

<figure>
  <a href="assets/architecture.svg"><img src="assets/architecture.svg" width="1024" height="690" alt="The fixed-domain decoder reads patch tokens. Each parameterized-domain query block reads boundary and domain tokens separately, averages the two messages, and updates the query state before time FiLM."></a>
  <figcaption>One patch read on the left. On the right, each spatial query block reads the two point streams separately and averages their messages before updating the query state.</figcaption>
</figure>

## Encoding irregular domains

The spinodal, channel and particle cases each use two streams. The boundary stream contains coordinates, normals and boundary type. The domain stream contains coordinates and a prescribed initial field. Boundary arc lengths and domain areas serve as quadrature measures. Invalid or padded entries carry zero measure and are masked in the fixed-shape payloads.

Each stream has its own MLP lift. It maps each point's position embedding and other features to one token independently. The encoder returns the two token arrays and their measures. Geometry parameters generate the physical point sets and known inputs used by these lifts.

## Queries and time

The spatial query starts from the embedding of the requested coordinate. In each cross-attention block, the same \(Q\) reads two contexts through separate cross-attention operations. The boundary and domain tokens supply separate \(K,V\) projections. The block averages the two messages, projects the mean and updates the query state through residual connections and an MLP. The next block reads the same two token arrays with the updated query state.

### Quadrature-weighted attention

For one head and one stream, let \(\kappa(\mathbf q,\mathbf k_j)\) be the dot product of the projected query and key after softmax over their feature dimensions. The head computes

$$
\mathcal A(\mathbf q)=
\frac{\displaystyle\sum_j m_j\,\kappa(\mathbf q,\mathbf k_j)\,\mathbf v_j}
     {\displaystyle\sum_j m_j\,\kappa(\mathbf q,\mathbf k_j)} .
$$

The fixed measure \(m_j\) represents the arc length or area assigned to the source point. At equal learned similarity, a point representing twice the measure contributes twice as much before normalization. A padded point has zero measure. Boundary and domain messages are normalized separately and then averaged in each block. The implementation contracts the weighted key–value moments before applying the query.

### Temporal decoding

Fixed domains can share a physical query set across conditions. Different geometries have their own coordinates, while the same geometry shares its spatial preparation across times. The temporal decoder uses feature-wise modulation (FiLM).

$$
\mathbf z_{\ell+1}
=\sigma\!\left(\boldsymbol\gamma_\ell(t)\odot
       W_\ell\mathbf z_\ell+\boldsymbol\beta_\ell(t)\right).
$$

This is the modulation pattern. The actual decoders also use their configured activations, embeddings and output maps. A time JVP through that map differentiates the resulting physical fields. Known fields can also appear in a hard lift, as with melting's initial signed distance. That use is distinct from query-local conditioning features in the learned attention path.

## Implementation

Read the [point conditional implementation](https://github.com/NanxiiChen/nonlocal-pinn/blob/main/src/nonlocal_pinn/nn/point_conditional.py) for the geometry encoder and see the [melting case](ice-melting.html) and [electropolish case](electropolish.html) for the patch configurations. [Training and validation](training.html) describes how the inputs are assembled and sampled.
