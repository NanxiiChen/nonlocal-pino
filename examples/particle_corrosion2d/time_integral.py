"""Global IVP balance; no time stepping or frozen learned predecessor."""
import jax.numpy as jnp


def simpson_queries(times):
    endpoints=jnp.sort(times)
    queries=jnp.zeros((2*len(times)-1,),dtype=times.dtype)
    queries=queries.at[::2].set(endpoints)
    queries=queries.at[1::2].set(.5*(endpoints[:-1]+endpoints[1:]))
    return endpoints,queries


def simpson_balance(endpoint_values,positive_rhs,initial,endpoints):
    """Return y(t)-y0+int G for y_t+G=0 at the positive endpoints.

    endpoint_values(Ncore,Nt),positive_rhs(Ncore,2Nt-1),initial(Ncore).
    The caller supplies the prescribed G(0),not an evolved reference value.
    """
    if (endpoint_values.ndim!=2 or positive_rhs.shape!=(len(initial),2*len(endpoints)-1)
            or endpoint_values.shape!=(len(initial),len(endpoints))):
        raise ValueError('Simpson balance requires aligned state/RHS/core/time arrays')
    widths=jnp.diff(endpoints)
    increments=(positive_rhs[:,0:-2:2]+4*positive_rhs[:,1:-1:2]+positive_rhs[:,2::2])*widths[None,:]/6
    return endpoint_values[:,1:]-initial[:,None]+jnp.cumsum(increments,axis=1)
