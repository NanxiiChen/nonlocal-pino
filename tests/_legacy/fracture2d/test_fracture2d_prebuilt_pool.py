"""Pooled and unpooled exact damage losses/gradients must agree."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from examples.fracture2d._legacy.configs.sigmoid_damage_load_refined import get_config
from examples.fracture2d._legacy.core_halo_damage import pack_rows
from examples.fracture2d._legacy.low_order_damage import make_loss_fn
from examples.fracture2d._legacy.plain_damage_head import fresh_network
from examples.fracture2d._legacy.precomputed_damage_sampling import make_sampling_rows,sample_core_halo
from examples.fracture2d._legacy.prebuilt_damage_pool import build_sampling_pool,make_pool_objective


class PrebuiltPoolTest(unittest.TestCase):
    def test_indexed_pool_matches_batch_loss_and_gradients_with_changed_history(self):
        jax.config.update('jax_enable_x64',True)
        c=get_config();c.arch.hidden_dim=4;c.arch.num_layers=1
        D=csr_matrix(np.array([[2.,-1.,-1.],[-1.,2.,-1.],[-1.,-1.,2.]]));B=csr_matrix(np.eye(3))
        bc,bv=map(jnp.asarray,pack_rows(B,width=1));net=fresh_network(c,2,B,head='sigmoid')
        rows=make_sampling_rows(D,row_width=3);cores=np.array([[2,0],[1,1],[0,2]],np.int32);prob=np.array([.1,.2,.7])
        pool,_=build_sampling_pool(rows,cores,prob,node_capacity=3);pool=jax.device_put(pool)
        X=jnp.array([[1.,2.],[3.,1.],[2.,4.]]);load=jnp.asarray(.005275)
        for norm in ['l1','l2']:
            _,batch_loss=make_loss_fn(gc0=112.5,gc1=.0648,gain=1.,population_size=3,norm=norm)
            pooled_loss=eqx.filter_jit(make_pool_objective(batch_loss,basis_columns=bc,basis_values=bv))
            for H in [jnp.array([1.,2.,4.]),jnp.array([2.,3.,5.])]:
                for slot,core in enumerate(cores):
                    packed=sample_core_halo(rows,core,node_capacity=3);ids=packed['global_ids']
                    ctx=dict(H=H,core_H=H[core],features=X[ids],basis_columns=bc[ids],basis_values=bv[ids],
                        core_local=jnp.asarray(packed['core_local']),operator_columns=jnp.asarray(packed['columns']),
                        operator_values=jnp.asarray(packed['values']),p_core=jnp.asarray(prob[core]))
                    pc=dict(H=H,features=X,pool=pool,slot=jnp.asarray(slot,jnp.int32))
                    np.testing.assert_allclose(pooled_loss(net,pc,load),batch_loss(net,ctx,load),rtol=1e-12)
                    a,b=eqx.filter_grad(pooled_loss)(net,pc,load),eqx.filter_grad(batch_loss)(net,ctx,load)
                    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_allclose(x,y,rtol=1e-11,atol=1e-12)


if __name__=='__main__':unittest.main()
