"""Phase/flow loss alternation with shared trunk and global SOAP state."""
import equinox as eqx
import jax.numpy as jnp


def stage_at(update,settings):
    """Zero-based host update index; joint tail keeps the coupled PDE active."""
    if update>=settings.alternating_steps:return 'joint',(True,True)
    if (update//settings.switch_every)%2==0:return 'phase',(True,False)
    return 'flow',(False,True)


def mask_output_updates(updates,active):
    """SOAP momentum/preconditioning must not move an inactive output row.

    Shared encoder/decoder parameters still update: this is loss/head staggering,
    not a claim that the full physical field is fixed during a block.
    """
    rows=jnp.stack((active[0],active[0],active[1],active[1]))
    if updates.output.weight.shape[0]!=4:
        raise ValueError('Partitioned head masking requires streamfunction4output model')
    return eqx.tree_at(lambda n:(n.output.weight,n.output.bias),updates,
        (updates.output.weight*rows[:,None],updates.output.bias*rows))
