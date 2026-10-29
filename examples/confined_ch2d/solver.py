"""Independent mixed-P2 FEM and regional SciPy reference; run in fenicsenv.

FEM uses the standard mixed CH weak form with BE-started BDF2 and Neumann
conditions natural. Native reference assembles its own SciPy matrix, no JAX
operator import. Both reference refinements must pass before train.py loads it.
"""
from pathlib import Path
import os,sys,argparse,json,importlib,time
if __package__ is None or __package__=='':sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ['PKG_CONFIG_PATH']=str(Path(sys.prefix)/'lib/pkgconfig')+os.pathsep+os.environ.get('PKG_CONFIG_PATH','')
# Existing environment's Python3.14/UFL error-format compatibility only.
import ufl.algorithms.check_arities as _arity
_original=_arity._afmt

def _format(items):
    if len(items)>0 and not isinstance(items[0],tuple):items=(items,)
    return _original(items) if items else ()
_arity._afmt=_format
import dolfin as df
import numpy as np
from examples.confined_ch2d.native_reference import native_matrix,native
from examples.confined_ch2d.problem import validate_config,initial_field,initial_laplacian,identity,relative
from examples.confined_ch2d.geometry import create_mesh,quadrature


def fem(c,h,dt,query,tag):
    root=Path(c.reference.root);root.mkdir(parents=True,exist_ok=True)
    meshpath=root/(tag+'.xdmf')
    vertices,cells=create_mesh(c,h,meshpath)
    mesh=df.Mesh()
    with df.XDMFFile(str(meshpath)) as file:file.read(mesh)
    elem=df.FiniteElement('Lagrange',mesh.ufl_cell(),2)
    W=df.FunctionSpace(mesh,elem*elem);V=df.FunctionSpace(mesh,elem)
    u,old=df.Function(W),df.Function(W)
    class IC(df.UserExpression):
        def eval(self,value,x):value[0]=float(initial_field(np.asarray(x)[None,:],c,xp=np)[0])
        def value_shape(self):return ()
    phi0=df.interpolate(IC(degree=6),V)
    mu0=df.Function(V)
    dofpoints=V.tabulate_dof_coordinates().reshape(-1,2)
    p0=initial_field(dofpoints,c,xp=np)
    mu0.vector().set_local(c.physical.beta*(p0**3-p0)-c.physical.epsilon**2*initial_laplacian(dofpoints,c))
    df.assign(u.sub(0),phi0);df.assign(u.sub(1),mu0);old.assign(u)
    older=df.Function(W);older.assign(u)
    phi,mu=df.split(u);phi_old,_=df.split(old);phi_older,_=df.split(older)
    v,w=df.TestFunctions(W);p=c.physical
    alpha,beta,gamma=df.Constant(1.),df.Constant(1.),df.Constant(0.)
    residual=(alpha*phi-beta*phi_old+gamma*phi_older)*v*df.dx+dt*p.mobility*df.dot(df.grad(mu),df.grad(v))*df.dx
    residual+=(mu-p.beta*(phi**3-phi))*w*df.dx-p.epsilon**2*df.dot(df.grad(phi),df.grad(w))*df.dx
    problem=df.NonlinearVariationalProblem(residual,u,[],df.derivative(residual,u))
    solver=df.NonlinearVariationalSolver(problem)
    prm=solver.parameters['newton_solver'];prm['linear_solver']='lu'
    prm['absolute_tolerance']=1e-10;prm['relative_tolerance']=1e-9;prm['maximum_iterations']=20
    df.set_log_level(40)
    if query is None:query,volumes=quadrature(vertices,cells)
    steps=round(p.terminal_time/dt);stride=steps//(c.reference.snapshots-1)
    fields={'phi':[],'mu':[],'mass':[],'energy':[]}
    start=time.perf_counter()
    def save():
        a,b=u.split(deepcopy=True)
        for name,f in [('phi',a),('mu',b)]:fields[name].append(np.array([f(df.Point(*x)) for x in query]))
        fields['mass'].append(float(df.assemble(a*df.dx)))
        fields['energy'].append(float(df.assemble((.25*p.beta*(a*a-1)**2+.5*p.epsilon**2*df.dot(df.grad(a),df.grad(a)))*df.dx)))
    save()
    # At t0 use the exact continuum IC derivatives at query points, not a noisy
    # discrete mass inversion of interpolated high-order spatial derivatives.
    initial=initial_field(query,c,xp=np)
    fields['mu'][0]=p.beta*(initial**3-initial)-p.epsilon**2*initial_laplacian(query,c)
    for i in range(1,steps+1):
        older.assign(old);old.assign(u)
        if i==2:alpha.assign(1.5);beta.assign(2.);gamma.assign(.5)
        solver.solve()
        if i%stride==0:save()
        if i%max(stride*5,1)==0:print(tag,i,'/',steps,flush=True)
    result={k:np.asarray(v) for k,v in fields.items()}
    result['seconds']=time.perf_counter()-start;result['vertices']=len(vertices);result['dofs']=W.dim()
    result['points']=query;result['mesh_points']=vertices;result['mesh_cells']=cells
    np.savez_compressed(root/(tag+'-fields.npz'),**result)
    return result


def run(c,preflight):
    validate_config(c)
    root=Path(c.reference.root);root.mkdir(parents=True,exist_ok=True)
    # Coarse mesh defines fixed physical centroid quadrature shared by both PINNs.
    meshpath=root/'collocation.xdmf'
    vertices,cells=create_mesh(c,c.geometry.mesh_size,meshpath)
    points,volumes=quadrature(vertices,cells)
    np.savez_compressed(root/'geometry.npz',points=points,volumes=volumes,mesh_points=vertices,mesh_cells=cells)
    print('collocation',len(points),'area',volumes.sum(),flush=True)
    h0,h1=c.reference.mesh_sizes
    coarse=fem(c,h0,c.reference.refined_dt,points,'fem_coarse')
    if preflight:
        np.savez_compressed(root/'preflight.npz',time=np.linspace(0,c.physical.terminal_time,c.reference.snapshots),
            points=points,volumes=volumes,mesh_points=vertices,mesh_cells=cells,**{k:coarse[k] for k in ('phi','mu','energy','mass')})
        print('Preflight completed; NOT a validated reference',flush=True);return
    fine_dt=fem(c,h1,c.reference.dt,points,'fem_fine_dt')
    fine=fem(c,h1,c.reference.refined_dt,points,'fem_fine')
    def errors(a,b):return {k:dict(pooled=relative(a[k],b[k],volumes),
        maximum_time=max(relative(x,y,volumes) for x,y in zip(a[k],b[k],strict=True))) for k in ('phi','mu')}
    checks=dict(local_space=errors(coarse,fine),local_time=errors(fine_dt,fine))
    A=native_matrix(points,volumes,c)
    first=native(c,points,volumes,A,c.reference.dt,1e-7)
    graph=native(c,points,volumes,A,c.reference.refined_dt,1e-9)
    checks['native_time']=errors(first,graph)
    checks['reference_gap']={k:relative(graph[k],fine[k],volumes) for k in ('phi','mu')}
    checks['local_mass_drift']=float(np.max(np.abs(fine['mass']-fine['mass'][0])))
    checks['native_mass_drift']=float(np.max(np.abs(graph['mass']-graph['mass'][0])))
    checks['local_energy_max_increase']=max(0.,float(np.max(np.diff(fine['energy']))))
    checks['native_energy_max_increase']=max(0.,float(np.max(np.diff(graph['energy']))))
    checks['resolution']={k:dict(vertices=v['vertices'],dofs=v['dofs'],seconds=v['seconds']) for k,v in [('coarse',coarse),('fine',fine)]}
    passed=True
    for label in ('local_space','local_time','native_time'):
        for key,tol in [('phi',c.reference.phase_tolerance),('mu',c.reference.chemical_tolerance)]:
            passed &= checks[label][key]['maximum_time']<tol
    passed &= checks['local_mass_drift']<1e-9 and checks['native_mass_drift']<1e-9
    passed &= checks['local_energy_max_increase']<1e-8 and checks['native_energy_max_increase']<1e-8
    checks['status']='passed' if passed else 'failed'
    output=dict(identity=json.dumps(identity(c)),points=points,volumes=volumes,mesh_points=vertices,mesh_cells=cells,
        time=np.linspace(0,c.physical.terminal_time,c.reference.snapshots))
    for k in ('phi','mu','mass','energy'):
        output[k+'_local']=fine[k];output[k+'_nonlocal']=graph[k]
    path=Path(c.reference.path);np.savez_compressed(path,**output)
    path.with_suffix('.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
    print(json.dumps(checks,indent=2),flush=True)
    if not passed:raise RuntimeError('Reference checks failed; do not train')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='examples.confined_ch2d.configs.train_coarsening_windows')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();run(importlib.import_module(args.config).get_config(),args.preflight)
