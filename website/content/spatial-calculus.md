For the regional diffusion cases, the loss evaluates a precomputed finite-horizon operator on predicted nodal fields. The channel uses a separate moment-corrected family for first and second spatial derivatives.

## Interior consistency

Consider a symmetric radial kernel supported on a ball of radius \(\delta\) around an interior point. Set \(\boldsymbol\xi=\mathbf y-\mathbf x\) and normalize its second moment to \(2I\).

$$
\int_{B_\delta(0)} K_\delta(\boldsymbol\xi)
  \,\xi_a\xi_b\,d\boldsymbol\xi = 2\,\delta_{ab}.
$$

Expanding the field at \(\mathbf x\) gives a linear term, a quadratic term and higher derivatives. The odd term integrates to zero on the full ball. The quadratic term then gives the negative Laplacian.

$$
\begin{aligned}
D_\delta u(\mathbf x)
 &= \int_{B_\delta(0)} K_\delta(\boldsymbol\xi)
        [u(\mathbf x)-u(\mathbf x+\boldsymbol\xi)]\,d\boldsymbol\xi \\
 &\simeq -\tfrac12\sum_{a,b}\partial_{ab}u(\mathbf x)
       \int_{B_\delta(0)}K_\delta(\boldsymbol\xi)\xi_a\xi_b\,d\boldsymbol\xi
  = -\Delta u(\mathbf x).
\end{aligned}
$$

The sign matters in the residual. For \(u_t=\nu\Delta u+s\), coordinate AD evaluates \(r_{\mathrm{AD}}=u_t-\nu\,\mathrm{tr}(\nabla_{\mathbf x}^2u)-s\). The regional graph instead evaluates \(r_{\mathrm{NL}}=u_t+\nu D_\delta u-s\) at its nodes. This replaces repeated spatial Hessian evaluation with neighborhood gathers and weighted sums. Consistency depends on the field, kernel, mesh and horizon. Training time and accuracy still require measured comparisons.

## Regional quadrature and walls

On a mesh with physical weights \(m_j\), the regional implementation uses neighbor differences. Visible-pair variants exclude a pair when the segment between its endpoints crosses an impermeable inclusion.

$$
(D_\delta u)_i=\sum_{j\in\mathcal N_\delta(i)}
  m_jK_{ij}(u_i-u_j),\qquad K_{ij}=K_{ji}\ge 0.
$$

<figure>
  <a href="assets/spatial-neighborhood.svg"><img src="assets/spatial-neighborhood.svg" width="960" height="344" alt="A finite-horizon neighborhood around a core point, with an inclusion blocking one potential edge"></a>
  <figcaption>The horizon selects nearby candidate points. A visibility rule can remove pairs across an impermeable inclusion.</figcaption>
</figure>

For a closed graph, every allowed interaction has an equal and opposite volume-weighted contribution at its endpoints. Thus \(D_\delta\mathbf 1=0\) and \(\sum_i m_i(D_\delta u)_i=0\). With nonnegative symmetric pairs, the discrete quadratic form is also nonnegative.

$$
\sum_i m_i u_i(D_\delta u)_i
=\tfrac12\sum_{i,j}m_i m_j K_{ij}(u_i-u_j)^2\ge0.
$$

This is a no-exchange condition for the regional dynamics. The full-ball Taylor argument changes near a truncated boundary. A finite-\(\delta\) field can therefore have a nonzero pointwise local normal derivative. Check the local Neumann limit separately from graph conservation, and handle prescribed Dirichlet or traction data through their own constraints. The graph identities apply to the discrete operator. Learned trajectories can still have residual error unless a quantity is also imposed as a hard constraint.

## Moment correction in the channel

At a channel target, build \(B\) from centered, horizon-scaled Taylor monomials of degrees one through three. Let \(a_0\) be the baseline radial derivative coefficients, \(W\) the diagonal kernel-quadrature weights and \(b\) the desired derivative moments. With sufficient local rank, the minimum weighted correction solves

$$
\min_a\ \lVert W^{-1/2}(a-a_0)\rVert_2^2
   \quad\text{subject to}\quad B^\mathsf T a=b.
$$

The closed-form correction is \(W B(B^\mathsf T W B)^{-1}(b-B^\mathsf T a_0)\). The code evaluates it by weighted QR and checks polynomial reproduction. Neighbor-minus-target differences cancel constants. The procedure builds derivative rows at interior and boundary targets for both gradient components and the **positive** Laplacian \(\Delta\).

The corrected rows need not be symmetric pairs. The channel prescribes wall and outlet residuals explicitly. AD is still used for the curl of its streamfunction, time JVP and parameter backpropagation. See the [stencil implementation](https://github.com/NanxiiChen/nonlocal-pino/blob/main/examples/channel_nsch2d/integral_operators.py) and the [channel case](two-phase-flow.html).
