"""Independent fully coupled mixed FEM reference. Run only in fenicsenv."""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys
import time

if __package__ is None or __package__=='':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ['PKG_CONFIG_PATH']=str(Path(sys.prefix)/'lib/pkgconfig')+os.pathsep+os.environ.get('PKG_CONFIG_PATH','')
import dolfin as df
import numpy as np
from examples.channel_nsch2d.geometry import random_walls,nominal_parameters,with_nominal,validate_mesh_queries,validate_parameters
from examples.channel_nsch2d.problem import walls,initial_field,flow_rate,material,identity,validate_config


def mesh_and_markers(c,amplitudes,nx,ny):
    g=c.geometry
    mesh=df.RectangleMesh(df.Point(0,0),df.Point(g.length,g.height),nx,ny)
    tags=df.MeshFunction('size_t',mesh,1,0)
    df.CompiledSubDomain('on_boundary && (near(x[1],0) || near(x[1],H))',H=g.height).mark(tags,1)
    df.CompiledSubDomain('on_boundary && near(x[0],0)').mark(tags,2)
    df.CompiledSubDomain('on_boundary && near(x[0],L)',L=g.length).mark(tags,3)
    coords=mesh.coordinates(); bottom,top=walls(coords[:,0],amplitudes,c,xp=np)
    coords[:,1]=bottom+coords[:,1]/g.height*(top-bottom)
    return mesh,tags


def fem(c,amplitudes,grid,dt,query,tag,output,*,common_query_grid=None,diagnostic_callback=None):
    """Solve the reference; optional read-only callback receives native snapshots."""
    validate_config(c)
    validate_parameters(amplitudes,c)
    if random_walls(c):np.testing.assert_array_equal(amplitudes,nominal_parameters(c))
    if len(grid)!=2 or any(isinstance(v,bool) or not isinstance(v,int) or v<4 for v in grid):
        raise ValueError('FEM grid requires two integer subdivision counts>=4')
    if not np.isfinite(dt) or dt<=0:raise ValueError('FEM dt must be positive finite')
    steps=c.physical.terminal_time/dt
    if abs(steps-round(steps))>1e-10 or round(steps)%(c.reference.snapshots-1):
        raise ValueError('FEM dt must divide the physical horizon and snapshot intervals')
    query=np.asarray(query,dtype=np.float64)
    if query.ndim!=2 or query.shape[1]!=2 or not np.isfinite(query).all():
        raise ValueError('FEM queries must be a finite(N,2) array')
    common_query=None
    if common_query_grid is not None:
        from examples.channel_nsch2d.response_queries import points,protocol
        common_query=points(c,common_query_grid)
        bottom,top=walls(common_query[:,0],amplitudes,c,xp=np)
        if np.any(common_query[:,1]<=bottom) or np.any(common_query[:,1]>=top):
            raise ValueError('Shared physical queries are outside this actual FEM geometry')
    all_queries=query if common_query is None else np.concatenate((query,common_query))
    validate_mesh_queries(all_queries,amplitudes,c,grid[0])
    if not df.has_lu_solver_method(c.reference.linear_solver):
        raise RuntimeError('Configured FEM linear solver is unavailable')
    mesh,tags=mesh_and_markers(c,amplitudes,*grid)
    cell=mesh.ufl_cell();P2=df.FiniteElement('Lagrange',cell,2)
    W=df.FunctionSpace(mesh,df.MixedElement([df.VectorElement('Lagrange',cell,2),
        df.FiniteElement('Lagrange',cell,1),P2,P2]))
    current,old,older=df.Function(W),df.Function(W),df.Function(W)
    velocity,pressure,phi,mu=df.split(current)
    test_velocity,test_pressure,test_phi,test_mu=df.TestFunctions(W)
    old_velocity,_,old_phi,_=df.split(old)
    older_velocity,_,older_phi,_=df.split(older)
    if c.reference.time_scheme!='bdf2':raise ValueError('Explicit reference time_scheme=bdf2 required')
    alpha,beta,gamma=df.Constant(1.),df.Constant(1.),df.Constant(0.)
    dx=df.Measure('dx',domain=mesh,metadata={'quadrature_degree':6})
    ds=df.Measure('ds',domain=mesh,subdomain_data=tags,metadata={'quadrature_degree':6})
    normal=df.FacetNormal(mesh);p=c.physical;A,kappa=material(c)
    inlet=df.Expression(('6*Q*x[1]*(H-x[1])/(H*H*H)','0'),Q=0.,H=c.geometry.height,degree=2)
    bc=[df.DirichletBC(W.sub(0),df.Constant((0.,0.)),tags,1),
        df.DirichletBC(W.sub(0),inlet,tags,2),df.DirichletBC(W.sub(2),df.Constant(1.),tags,2),
        df.DirichletBC(W.sub(3),df.Constant(0.),tags,2)]
    residual=p.density*df.inner((alpha*velocity-beta*old_velocity+gamma*older_velocity)/dt,test_velocity)*dx
    residual+=p.density*df.inner(df.grad(velocity)*velocity,test_velocity)*dx
    residual+=2*p.viscosity*df.inner(df.sym(df.grad(velocity)),df.sym(df.grad(test_velocity)))*dx
    residual+=(-pressure*df.div(test_velocity)+test_pressure*df.div(velocity))*dx
    residual+=df.inner(phi*df.grad(mu),test_velocity)*dx
    residual+=(alpha*phi-beta*old_phi+gamma*older_phi)/dt*test_phi*dx-phi*df.dot(velocity,df.grad(test_phi))*dx
    residual+=p.mobility*df.dot(df.grad(mu),df.grad(test_phi))*dx
    residual+=phi*df.dot(velocity,normal)*test_phi*ds(3)
    residual+=(mu-A*(phi**3-phi))*test_mu*dx-kappa*df.dot(df.grad(phi),df.grad(test_mu))*dx
    problem=df.NonlinearVariationalProblem(residual,current,bc,df.derivative(residual,current))
    solver=df.NonlinearVariationalSolver(problem)
    prm=solver.parameters['newton_solver'];prm['linear_solver']=c.reference.linear_solver
    prm['absolute_tolerance']=c.reference.newton_tolerance
    prm['relative_tolerance']=c.reference.newton_tolerance;prm['maximum_iterations']=c.reference.newton_iterations
    V=df.FunctionSpace(mesh,P2)
    class Initial(df.UserExpression):
        def eval(self,values,x):values[0]=float(initial_field(np.asarray(x)[None,:],c,xp=np)[0])
        def value_shape(self):return ()
    initial=df.interpolate(Initial(degree=6),V)
    df.assign(current.sub(2),initial)
    trial,test=df.TrialFunction(V),df.TestFunction(V);mu0=df.Function(V)
    df.solve(trial*test*dx==(A*(initial**3-initial)*test+kappa*df.dot(df.grad(initial),df.grad(test)))*dx,
        mu0,df.DirichletBC(V,df.Constant(0.),tags,2),solver_parameters={'linear_solver':c.reference.linear_solver})
    df.assign(current.sub(3),mu0);old.assign(current);older.assign(current)
    steps=round(p.terminal_time/dt);stride=steps//(c.reference.snapshots-1)
    fields=[];common_fields=[];diagnostics=[];iterations=[];start=time.perf_counter()
    # Coordinates lie in every piecewise-linear mesh by construction when the
    # evaluation points use the same normalized tensor grid or interior offset.
    def save(t):
        u,pr,ph,m=current.split(deepcopy=True)
        rows=[]
        for point in all_queries:
            x=df.Point(*point);v=u(x);rows.append((ph(x),m(x),v[0],v[1],pr(x)))
        values=np.asarray(rows);fields.append(values[:len(query)])
        if common_query is not None:common_fields.append(values[len(query):])
        diagnostics.append(dict(time=t,dispersed_area=float(df.assemble((1-ph)/2*dx)),
            inlet_flux=float(-df.assemble(df.dot(u,normal)*ds(2))),
            outlet_flux=float(df.assemble(df.dot(u,normal)*ds(3))),
            divergence_l2=float(np.sqrt(df.assemble(df.div(u)**2*dx))),
            wall_speed_l2=float(np.sqrt(df.assemble(df.inner(u,u)*ds(1)))),
            energy=float(df.assemble((.5*p.density*df.inner(u,u)+A/4*(ph*ph-1)**2+
                kappa/2*df.inner(df.grad(ph),df.grad(ph)))*dx))))
        if diagnostic_callback is not None:
            diagnostic_callback(t,(u,pr,ph,m),mesh,tags)
    save(0.)
    for step in range(1,steps+1):
        older.assign(old);old.assign(current);inlet.Q=float(flow_rate(step*dt,c,xp=np))
        if step==2:alpha.assign(1.5);beta.assign(2.);gamma.assign(.5)
        count,converged=solver.solve()
        if not converged:raise RuntimeError(f'{tag} Newton failed at step {step}')
        iterations.append(count)
        if step%stride==0:
            save(step*dt)
            print(f'{tag}: {step}/{steps}, Newton={count}, area={diagnostics[-1]["dispersed_area"]:.8f}',flush=True)
    result=dict(fields=np.asarray(fields),points=query,amplitudes=np.asarray(amplitudes),
        time=np.linspace(0,p.terminal_time,c.reference.snapshots),
        mesh_points=mesh.coordinates(),mesh_cells=mesh.cells(),identity_json=json.dumps(identity(c)))
    if common_query is not None:
        result.update(common_fields=np.asarray(common_fields),common_points=common_query,
            common_query_grid=np.asarray(common_query_grid,dtype=np.int64),
            common_query_identity=json.dumps(protocol(c,common_query_grid)))
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**result)
    report=dict(status='completed_not_refinement_validated',seconds=time.perf_counter()-start,
        grid=list(grid),dt=dt,dofs=W.dim(),amplitudes=list(amplitudes),time_scheme=c.reference.time_scheme,
        diagnostics=diagnostics,maximum_newton_iterations=max(iterations),diagnostic_queries=len(query),
        formulation='Full implicit matched-property NS-CH; zero hydrodynamic outlet traction')
    if common_query is not None:report['common_query_count']=len(common_query)
    path.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return result,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    configuration=parser.add_mutually_exclusive_group()
    configuration.add_argument('--config',default='examples.channel_nsch2d.configs.train_base')
    configuration.add_argument('--saved-config',help='Explicit saved config JSON; physical parameters remain frozen')
    parser.add_argument('--grid',nargs=2,type=int,required=True)
    parser.add_argument('--dt',type=float,required=True)
    parser.add_argument('--query-grid',nargs=2,type=int,required=True)
    parser.add_argument('--common-query-grid',nargs=2,type=int,help='Additional fixed physical queries in the common prior subdomain')
    parser.add_argument('--upper',type=float);parser.add_argument('--lower',type=float)
    parser.add_argument('--parameters',nargs='+',type=float,help='Explicit full random-wall coefficient vector; absent uses declared nominal')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.saved_config is None:
        c=importlib.import_module(args.config).get_config()
    else:
        from ml_collections import ConfigDict
        c=ConfigDict(json.loads(Path(args.saved_config).read_text()))
    if random_walls(c):
        if args.upper is not None or args.lower is not None:raise ValueError('Spline walls require --parameters, not two amplitudes')
        a=nominal_parameters(c) if args.parameters is None else args.parameters
    else:
        if args.parameters is not None or args.upper is None or args.lower is None:raise ValueError('Legacy walls require both --upper and --lower')
        a=(args.upper,args.lower)
    c=with_nominal(c,a);validate_config(c)
    # Interior diagnostic queries avoid chord-versus-exact-wall ambiguity when
    # comparing two different curved-boundary FE resolutions.
    qx,qy=args.query_grid
    if min(qx,qy)<1:raise ValueError('Query-grid counts must be positive')
    x=(np.arange(qx)+.5)*c.geometry.length/qx
    e=(np.arange(qy)+.5)/qy
    xx,ee=np.meshgrid(x,e,indexing='ij');b,t=walls(xx,a,c,xp=np)
    query=np.stack((xx,b+ee*(t-b)),axis=-1).reshape(-1,2)
    df.set_log_level(40)
    common=None if args.common_query_grid is None else tuple(args.common_query_grid)
    _,report=fem(c,a,tuple(args.grid),args.dt,query,'fem',args.output,common_query_grid=common)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
