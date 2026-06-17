"""Data-driven diagnostic: full MLP fitting FEM solution."""
import logging, os, time
import jax.numpy as jnp, jax.random as jr, optax, equinox as eqx, numpy as np
from tensorboardX import SummaryWriter
from pinn.arch import make_pinn
from pinn.operators import build_D

import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "losses_full", os.path.join(os.path.dirname(__file__), 'losses_full.py'))
_lf = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_lf)
_compute_all = _lf._compute_all
loss_ac = _lf.loss_ac
loss_ch = _lf.loss_ch

log = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    rng = jr.PRNGKey(0)
    k_net, = jr.split(rng, 1)

    fem = np.load('/root/autodl-tmp/data/corrosion1d/results_diffusion.npz')
    x_fem = jnp.array(fem['mesh'])[:, None]
    t_fem = jnp.array(fem['time'])
    phi_fem = jnp.array(fem['phi']).T
    c_fem = jnp.array(fem['c']).T
    log.info("FEM: %d x, %d t", phi_fem.shape[0], phi_fem.shape[1])

    net = make_pinn(spatial_dim=1, out_dim=2, hidden_dim=64, num_layers=4, key=k_net)

    D_phys, A, L = 8.5e-10, 5.35e7, 2.0
    a_phi, w_phi = 1.03e-4, 1.76e7
    c_se, c_le = 1.0, 5100.0/1.43e5
    L_c, T_c = 100e-6, 50.0
    dc0 = c_se - c_le
    nd = (D_phys*T_c/L_c**2, 2.*A*L*T_c, L*w_phi*T_c, L*a_phi*T_c/L_c**2, dc0, c_le)

    D = build_D(x_fem, 0.05, kernel="cubic")

    run_dir = f"/root/autodl-tmp/tf-logs/corrosion1d/diag_fullmlp_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    log.info("Logging to %s", run_dir)

    opt = optax.adam(5e-4)
    opt_state = opt.init(eqx.filter(net, eqx.is_array))

    @eqx.filter_jit
    def step(net, opt_state):
        (loss, grad) = eqx.filter_value_and_grad(
            lambda n: jnp.mean((_compute_all(n, x_fem, t_fem)[:2][0] - phi_fem)**2
                               + (_compute_all(n, x_fem, t_fem)[:2][1] - c_fem)**2))(net)
        updates, opt_state = opt.update(grad, opt_state)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss

    for it in range(1, 5001):
        net, opt_state, loss_val = step(net, opt_state)
        if it % 200 == 0:
            phi_p, c_p, _, _ = _compute_all(net, x_fem, t_fem)
            rel_phi = float(jnp.sqrt(jnp.mean((phi_p-phi_fem)**2))/jnp.sqrt(jnp.mean(phi_fem**2)))
            rel_c = float(jnp.sqrt(jnp.mean((c_p-c_fem)**2))/jnp.sqrt(jnp.mean(c_fem**2)))
            ac = loss_ac(net, x_fem, t_fem, D, nd)
            ch = loss_ch(net, x_fem, t_fem, D, nd)
            writer.add_scalar('data/loss', float(loss_val), it)
            writer.add_scalar('eval/rel_L2_phi', rel_phi, it)
            writer.add_scalar('eval/rel_L2_c', rel_c, it)
            writer.add_scalar('pde_mon/ac', float(ac), it)
            writer.add_scalar('pde_mon/ch', float(ch), it)
            log.info("[%5d] loss=%.3e rel-phi=%.4f rel-c=%.4f pde:ac=%.3e ch=%.3e",
                     it, float(loss_val), rel_phi, rel_c, float(ac), float(ch))
        if it % 1000 == 0:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            phi_ev, c_ev, _, _ = _compute_all(net, x_fem, t_fem)
            err_phi, err_c = phi_ev-phi_fem, c_ev-c_fem
            cmap='coolwarm'; extent=[0.,1.,float(t_fem[-1]),float(t_fem[0])]
            fig, axes = plt.subplots(2,2,figsize=(12,8))
            for i,(d,t,vm) in enumerate([(phi_ev.T,r'$\phi$ pred',(0,1)),
                (err_phi.T,r'$\phi$ err',None),(c_ev.T,r'$c$ pred',(0,1)),
                (err_c.T,r'$c$ err',None)]):
                ax=axes[i//2,i%2]
                im=ax.imshow(d,cmap=cmap,aspect='auto',origin='upper',
                    extent=extent,interpolation='bilinear',
                    vmin=vm[0] if vm else None,vmax=vm[1] if vm else None,rasterized=True)
                ax.set_title(t); ax.set_xlabel(r'$x^*$'); ax.set_ylabel(r'$t^*$')
                plt.colorbar(im,ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir,f'eval_{it:06d}.png'),dpi=150)
            writer.add_figure('eval/heatmaps',fig,it); plt.close(fig)
    writer.close()
    log.info("Done. Logs at %s", run_dir)
