The five operators share a coordinate-query decoder pattern. The condition encoder depends on whether the input is a field on a fixed grid or a domain with sample-dependent geometry.

## Encoding fixed domains

Melting conditions on a \(32^3\) initial phase volume. A nonoverlapping \(4^3\) patch yields one token, giving \(8^3=512\) tokens. Electropolish uses \(4\times8\) patches of a \(32\times64\) image, giving \(8\times8=64\) tokens. Project flattened patches into the model width, retain patch positions, and mix tokens with self-attention. Both selected models use two condition blocks and four heads.

<figure>
  <a href="assets/architecture.svg"><img src="assets/architecture.png" width="1024" height="904" alt="Master figure: background geometry and operator preparation, rolling input pools, core-halo sampling, patch or two-stream point encoders, spatial cross-attention, three time-FiLM decoder layers, and nonlocal physics training."></a>
  <figcaption>Architecture and training workflow. Patch encoding and two-stream point encoding are alternatives; both prepare spatial features before per-layer time FiLM. The lower panels summarize nonlocal residual evaluation and case-dependent training strategies. Field textures are schematic. <a href="assets/architecture.pdf">Editable PDF</a> · <a href="assets/architecture.svg">SVG</a>.</figcaption>
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

Fixed domains can share a physical query set across conditions. Different geometries have their own coordinates. For a given condition and spatial query set, the same prepared features initialize the decoder at every sampled time. A time MLP takes the time embedding and produces separate scale and shift vectors \((\boldsymbol\gamma_\ell(t),\boldsymbol\beta_\ell(t))\) for each of the three hidden layers. All spatial queries at the same time share these vectors.

$$
\mathbf z_\ell
=\operatorname{GELU}\!\left[
   \bigl(1+\boldsymbol\gamma_\ell(t)\bigr)\odot
   \bigl(W_\ell\mathbf z_{\ell-1}+\mathbf b_\ell\bigr)
   +\boldsymbol\beta_\ell(t)\right],\qquad \ell=1,2,3.
$$

The linear output head reads \(\mathbf z_3\), and each case applies its physical output map. A time JVP through that map differentiates the resulting physical fields. Known fields can also appear in a hard lift, as with melting's initial signed distance. That use is distinct from query-local conditioning features in the learned attention path.

## Reading the training panels

The top panel describes known-input preparation. Geometry workers construct
meshes, quadrature and operators, then replace entries in a bounded rolling
pool. Fixed-domain cases draw fresh initial fields and reuse the fixed-domain
operator; they do not remesh for each field. Core/halo sampling retains every
neighbor needed by a selected residual row. It is an available sampling path,
not a claim that all selected cases use spatial minibatches.

In the geometry panel, the two contexts remain separate through their MLP
lifts and weighted attention reads. The shared query starts from a Fourier
embedding of the spatial coordinate; each block updates it with the projected
mean of both messages, residual connections and an MLP. Arc-length and area
weights normalize their respective streams. Time enters the subsequent MLP
decoder, with a separate scale/shift pair at each hidden layer.

The nonlocal panel illustrates a Laplacian stencil: the primed value belongs
to a neighboring point, and each weight includes its integration measure.
The positive operator shown approximates minus the Laplacian. Domain-restricted
neighborhoods implement the case's finite-horizon wall treatment; this is not
an exact, universal enforcement of every Neumann condition. The channel also
uses moment-corrected gradient stencils and retains an AD streamfunction curl.
Time JVP and backpropagation to network parameters remain part of training.

Residual normalization/preconditioning, fixed or inverse-gradient-norm
weighting, and Adam/SOAP are selected per case. Causal weighting is available
but disabled in the five selected recipes. Initial/boundary conditions use
their documented hard lifts or soft losses; a hard-imposed term is not added
again as a duplicate penalty. The diagram is a schematic of these mechanisms,
not one combined preset. See [training and validation](training.html).

## Source implementations

Read the [point conditional implementation](https://github.com/NanxiiChen/nonlocal-pino/blob/main/src/nonlocal_pinn/nn/point_conditional.py) for the geometry encoder and see the [melting case](ice-melting.html) and [electropolish case](electropolish.html) for the patch configurations. [Training and validation](training.html) describes how the inputs are assembled and sampled.
