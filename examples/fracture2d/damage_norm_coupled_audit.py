"""Native coupled L16 replay and evaluation-only crack-path diagnostics."""
import argparse,json,os,time
from pathlib import Path
import jax
from ml_collections import ConfigDict
import numpy as np
from scipy.sparse import coo_matrix,load_npz
from matplotlib.tri import Triangulation,LinearTriInterpolator
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.local_damage_trial import load_geometry,sha
from examples.fracture2d.staggered_damage_audit import native_image,read_fields,residual


def peak_profile(d,tri,*,x_grid,y_grid,threshold):
    """P1 transverse peak and contiguous half-height width on declared cuts."""
    X,Y=np.meshgrid(x_grid,y_grid,indexing='ij')
    values=np.asarray(LinearTriInterpolator(tri,d)(X,Y))
    if not np.isfinite(values).all():raise ValueError('path cuts leave valid material domain')
    peaks=values.argmax(axis=1);maxima=values.max(axis=1);active=maxima>=threshold
    x,y=x_grid[active],y_grid[peaks[active]];widths=[];censored=[]
    for row,peak in zip(values[active],peaks[active]):
        left=right=int(peak);half=row[peak]/2
        while left>0 and row[left-1]>=half:left-=1
        while right<len(y_grid)-1 and row[right+1]>=half:right+=1
        widths.append(float(y_grid[right]-y_grid[left]));censored.append(left==0 or right==len(y_grid)-1)
    drift=y-.5
    return dict(x=x.tolist(),y_at_peak=y.tolist(),peak_d=maxima[active].tolist(),
        half_height_width=widths,width_censored=censored,
        max_abs_lateral_drift=float(abs(drift).max()) if len(x) else None,
        rms_lateral_drift=float(np.sqrt(np.mean(drift**2))) if len(x) else None,
        active_cuts=len(x),threshold=threshold,method='P1 transverse max; contiguous half-height band',
        resolution=float(y_grid[1]-y_grid[0]),matched_extent_required=True)


def main(root,*,watch):
    if jax.devices()[0].platform!='cpu':raise ValueError('coupled norm replay requires CPU')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    while not (root/'preflight.json').exists():
        if (root/'exit.json').exists():raise RuntimeError('coupled norm exits before preflight')
        if not watch:raise RuntimeError('coupled norm preflight unavailable')
        time.sleep(5)
    identity=json.loads((root/'identity.json').read_text());a=ConfigDict(identity['config'],allow_dotted_keys=True)
    c=ConfigDict(identity['source_config'],allow_dotted_keys=True);t=a.coupled_training
    for source,pins in [(Path(a.source_root),a.pins),(Path(a.norm_trial.screen_source),a.norm_trial.screen_pins),(Path(t.screen_root),t.screen_pins)]:
        for name,digest in pins.items():assert sha(source/name)==digest,name
    for name,digest in identity['code'].items():assert sha(Path('/root/nonlocal-KKS')/name)==digest,name
    m,op,geometry=load_geometry(c);assert geometry==identity['geometry']
    D,B=op.to_scipy(),load_npz(c.l1_loading.basis_path);scales=read_fields(Path(a.source_root)/'parameterization.npz')
    parent=read_fields(Path(a.source_root)/f'state_{a.parent_index:04d}/fields.npz')
    g0,g1=c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0;xy=m.coordinates[:m.n_material]
    tri,area=m.triangles[:m.n_material_cells],m.areas[:m.n_material_cells]
    rows=np.broadcast_to(tri[:,:,None],(len(tri),3,3)).ravel();cols=np.broadcast_to(tri[:,None,:],(len(tri),3,3)).ravel()
    mass=coo_matrix(((area[:,None,None]*(np.ones((3,3))+np.eye(3))/12).ravel(),(rows,cols)),shape=(m.n_material,m.n_material)).tocsr()
    def rms(v):return float(np.sqrt(np.sum(v*(mass@v))/area.sum()))
    triangulation=Triangulation(xy[:,0],xy[:,1],tri)
    ref=read_fields(Path('/root/autodl-tmp/tf-logs/fracture2d/staggered_damage_trial_20261013/direct_full/final/fields.npz'))
    np.testing.assert_array_equal(ref['H_parent'],parent['H'])
    review=root/'independent_review';review.mkdir(exist_ok=True);done=set();audited=[]
    while True:
        paths=sorted((root/'pinn_full').glob('outer_[0-9][0-9][0-9][0-9]'))+[root/'pinn_full/final']
        for folder in paths:
            rp=folder/'report.json'
            if not rp.exists() or str(rp) in done:continue
            r,f=json.loads(rp.read_text()),read_fields(folder/'fields.npz');final=folder.name=='final'
            assert sha(folder/'fields.npz')==r['fields_sha256'] and sha(folder/'network.eqx')==r['network_sha256']
            assert r['loss_norm']=='l16' and r['optimizer']=='lbfgs' and r['omega']==1. and r['load']==a.load
            for name,v in f.items():assert v.dtype==np.float64 and np.isfinite(v).all(),name
            np.testing.assert_array_equal(f['H_parent'],parent['H']);np.testing.assert_array_equal(f['d_parent'],parent['d'])
            np.testing.assert_array_equal(f['d'],f['damage_image'])
            stage_q=f['q_stage'] if final else f['q']
            np.testing.assert_array_equal(f['H_used'],np.maximum(parent['H'],m.driver(stage_q)))
            np.testing.assert_array_equal(f['H'],np.maximum(parent['H'],m.driver(f['q'])))
            np.testing.assert_array_equal(f['q'][m.constrained],(a.load*m.load_direction)[m.constrained])
            d=native_image(folder/'network.eqx',f['H_used'],c,D,B,xy,scales['column_scale'],float(scales['trunk_gain']),g0)
            np.testing.assert_allclose(d,f['damage_image'],rtol=1e-9,atol=1e-11)
            raw=(g0+2*f['H_used'])*d+g1*(D@d)-2*f['H_used'];rho=raw/(2*np.sqrt(np.mean(f['H_used']**2)))
            scale=max(float(abs(rho).max()),t.floor)
            loss=float(scale**2*(np.mean((rho/scale)**t.power)+(t.floor/scale)**t.power)**(2/t.power))
            np.testing.assert_allclose(loss,r['full_loss'],rtol=1e-9,atol=1e-12)
            inner=residual(d,f['H_used'],D,g0,g1);np.testing.assert_allclose(inner,r['inner_AT2'],rtol=1e-9,atol=1e-12)
            np.testing.assert_allclose(rms((stage_q-f['q_previous']).reshape(-1,2)[:m.n_material])/a.load,r['displacement_rms_over_load'],rtol=1e-9,atol=1e-12)
            np.testing.assert_allclose(rms(d-f['d_previous']),r['damage_rms'],rtol=1e-9,atol=1e-12)
            force=m.matrix(f['d'] if final else f['d_previous'])@f['q'];norm=float(np.linalg.norm(force[m.free]))
            assert norm<=r['mechanics']['convergence_limit']*(1+1e-3)
            np.testing.assert_allclose(force@m.load_direction,r['reaction'],rtol=1e-9,atol=1e-6)
            assert r['mechanics_calls']==r['sweep']+int(final) and r['optimizer_count'] is None
            journal=[json.loads((root/'pinn_full'/f'sweep_{i:04d}.json').read_text()) for i in range(1,r['sweep']+1)]
            assert sum(x['optimization']['iterations'] for x in journal)==r['training_updates']
            assert sum(x['optimization']['evaluations'] for x in journal)==r['function_evaluations']
            assert all(0<=x['optimization']['iterations']<=t.inner_updates for x in journal)
            profile=peak_profile(d,triangulation,x_grid=np.linspace(.52,.98,47),y_grid=np.linspace(.4,.6,401),threshold=.5)
            row=dict(sweep=r['sweep'],final=final,pass_replay=True,reaction=r['reaction'],inner_AT2=inner,
                host_force_norm=norm,full_loss=loss,uniform_L1=float(abs(rho).mean()),
                damage_L2_vs_final_direct=float(np.sqrt(m.volumes@(d-ref['d'])**2/(m.volumes@ref['d']**2))),
                peak_profile=profile,fields_sha256=r['fields_sha256'],report_sha256=sha(rp))
            if final:
                current=native_image(folder/'network.eqx',f['H'],c,D,B,xy,scales['column_scale'],float(scales['trunk_gain']),g0)
                np.testing.assert_allclose(current,f['raw_current_image'],rtol=1e-9,atol=1e-11)
                physical=residual(d,f['H'],D,g0,g1);np.testing.assert_allclose(physical,r['physical_AT2'],rtol=1e-9,atol=1e-12)
                assert not r['load_advanced'] and not r['certified_path'];row['physical_AT2']=physical
            audited.append(row);done.add(str(rp));print(json.dumps(dict(event='coupled_L16_CPU',sweep=r['sweep'],final=final,pass_replay=True,active_cuts=profile['active_cuts'])),flush=True)
        finished=(root/'result.json').exists() and json.loads((root/'result.json').read_text())['complete']
        complete=finished and any(x['final'] for x in audited)
        atomic_json(review/'audit.json',dict(complete=complete,pass_replay=True,rows=audited,morphology_primary=True,
            no_load_advancement=True,auditor_sha256=sha(__file__)))
        if complete or not watch:return
        if (root/'exit.json').exists() and json.loads((root/'exit.json').read_text())['worker_exit_code']!=0:raise RuntimeError('coupled norm worker fails')
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    jax.config.update('jax_enable_x64',True);main(Path(args.root),watch=args.watch)
