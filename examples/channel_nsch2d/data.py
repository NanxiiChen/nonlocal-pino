"""Geometry-only physical payloads; no evolved training labels."""
import numpy as np
from examples.channel_nsch2d.problem import boundary,boundary_loss_quadrature,quadrature,initial_field
from examples.channel_nsch2d.geometry import validate_parameters,initial_parameters
from examples.channel_nsch2d.operators import laplacian_stencils,first_derivative_stencils


def geometry_data(amplitudes,c):
    if c.nl.method == 'fem_weak_p2p1':
        from examples.channel_nsch2d.fem_weak import geometry_data as finite_element_data
        return finite_element_data(amplitudes, c)
    a=validate_parameters(amplitudes,c)
    points,volumes=quadrature(a,c,c.sample.Nx,c.sample.Ny)
    integral=c.nl.method=='integral_moment'
    if integral:
        from examples.channel_nsch2d.integral_operators import integral_stencils
        indices,gx,gy,lap,report=integral_stencils(points,volumes,points,c)
    else:
        indices,lap,report=laplacian_stencils(points,c)
    b,bw=boundary(a,c)
    interior=np.ones((c.sample.Nx+1,c.sample.Ny+1));interior[[0,-1],:]=0;interior[:,[0,-1]]=0
    data=dict(points=points,volumes=volumes,neighbors=indices,laplacian=lap,
        interior=interior.ravel(),amplitudes=a,boundary=b,boundary_weights=bw,
        domain=np.column_stack((points,initial_field(points,c,xp=np))),domain_weights=volumes)
    if 'boundary_sampling' in c:
        data.update(encoder_boundary=b,encoder_boundary_weights=bw)
        b,bw=boundary_loss_quadrature(a,c)
        data.update(boundary=b,boundary_weights=bw)
        report['boundary_sampling']=dict(**c.boundary_sampling.to_dict(),
            points=len(b),encoder_points=len(data['encoder_boundary']),
            nearest_wall_endpoint=float(np.min(np.minimum(b[:2*c.boundary_sampling.points_per_side,0],
                c.geometry.length-b[:2*c.boundary_sampling.points_per_side,0]))))
    if 'derivatives' in c:
        if integral:
            gr=report.copy()
            bi,bx,by,_,br=integral_stencils(points,volumes,b[:,:2],c)
        else:
            adjacent,gx,gy,gr=first_derivative_stencils(points,points,c)
            np.testing.assert_array_equal(adjacent,indices)
            bi,bx,by,br=first_derivative_stencils(points,b[:,:2],c)
        data.update(gradient_x=gx,gradient_y=gy,boundary_neighbors=bi,
            boundary_gradient_x=bx,boundary_gradient_y=by)
        report['first_derivatives']=dict(domain=gr,boundary=br,
            scope=('Integral PDE/boundary derivatives; physical streamfunction curl remains AD'if integral else
                'GMLS PDE/boundary first derivatives; physical streamfunction curl remains AD'))
        if integral:
            report['maximum_condition']=max(report['maximum_condition'],br['maximum_condition'])
            report['polynomial_error']=max(report['polynomial_error'],br['polynomial_error'])
    if 'phase_transport' in c:
        from examples.channel_nsch2d.transport import build_transport
        data.update(build_transport(points,a,c))
        report['phase_transport']=dict(kind=c.phase_transport.kind,
            measure=float(data['transport_measure'].sum()),minimum_height=float(data['transport_jacobian'].min()))
    return data,report


def build_bank(c):
    amplitudes=initial_parameters(c)
    data=[];reports=[]
    for a in amplitudes:
        d,r=geometry_data(a,c);data.append(d);reports.append(r)
    return {k:np.stack([d[k] for d in data]) for k in data[0]},reports
