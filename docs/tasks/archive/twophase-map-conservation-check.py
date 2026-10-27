"""Algebra/zero-mode illustration for the explanation; no training or model edit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from examples.twophase2d.configs.train_vortex_conserved import get_config
from examples.twophase2d.solver import symbols


def inverse(u):
    return 2/np.sqrt(3)*np.sinh(np.arcsinh(1.5*np.sqrt(3)*u)/3)


def main():
    phi=np.linspace(-1.5,1.5,301)
    u=phi+phi**3
    eps=1e-5
    inverse_error=float(np.max(np.abs(inverse(u)-phi)))
    derivative_error=float(np.max(np.abs((inverse(u+eps)-inverse(u-eps))/(2*eps)-1/(1+3*phi**2))))
    F=.25*(phi*phi-1)**2
    Fc=.25*phi**4+.5*phi**2
    convex_split_error=float(np.max(np.abs(F-(Fc-phi**2+.25))))
    assert inverse_error<1e-14 and derivative_error<1e-9 and convex_split_error<1e-14
    c=get_config()
    n=c.sample.Nx
    rng=np.random.default_rng(7)
    p=rng.normal(size=(n,n))
    flux=rng.normal(size=(n,n,2))
    k=2*np.pi*np.fft.fftfreq(n,d=c.geometry.length/n)
    k[n//2]=0
    hat=np.fft.rfft2(flux,axes=(0,1))
    adv=np.fft.irfft2(1j*(k[None,:n//2+1]*hat[...,0]+k[:,None]*hat[...,1]),s=(n,n))
    lam,_=symbols(n,c,'nonlocal')
    diffusion=np.fft.ifft2(lam*np.fft.fft2(p)).real
    p0=rng.normal(size=(n,n))
    projected=p-np.mean(p-p0)
    flux_mean=float(abs(adv.mean()))
    diffusion_mean=float(abs(diffusion.mean()))
    mass_error=float(abs(projected.mean()-p0.mean()))
    assert flux_mean<1e-11 and diffusion_mean<1e-10 and mass_error<1e-14
    # Illustration of unresolved continuum slopes versus nodal differentiation.
    x=np.arange(n)/n
    nodal=np.sin(2*np.pi*n*x)
    continuum_slope=2*np.pi*n*np.cos(2*np.pi*n*x)
    nodal_slope=np.fft.ifft(1j*k*np.fft.fft(nodal)).real
    reference=Path('/root/autodl-tmp/data/twophase2d/results_periodic_vortex.npz')
    with np.load(reference,allow_pickle=False) as f:
        ranges={key:[float(f[key].min()),float(f[key].max())] for key in ('phi_local','phi_nonlocal')}
    record=dict(status='passed',inverse_error=inverse_error,inverse_derivative_error=derivative_error,
        convex_split_error=convex_split_error,flux_zero_mode_error=flux_mean,
        nonlocal_diffusion_zero_mode_error=diffusion_mean,mass_projection_error=mass_error,
        unresolved_wave_illustration=dict(max_nodal_value=float(np.abs(nodal).max()),
            mean_correct_continuum_slope=float(continuum_slope.mean()),
            mean_nodal_spectral_slope=float(nodal_slope.mean())),reference_ranges=ranges,
        scope='Algebra and discrete zero modes only; no conditioning or training-benefit theorem.')
    output=Path(__file__).parent
    (output/'twophase-map-conservation-checks.json').write_text(json.dumps(record,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    raw=np.linspace(-3,3,601)
    axes[0].plot(raw,raw,label='Linear')
    axes[0].plot(raw,np.tanh(raw),label='tanh')
    axes[0].plot(raw,inverse(raw),label='Inverse cubic')
    axes[0].axhline(1,color='gray',linestyle=':',linewidth=.8)
    axes[0].axhline(-1,color='gray',linestyle=':',linewidth=.8)
    axes[0].set_ylim(-1.6,1.6)
    axes[0].set_xlabel('Network coordinate z');axes[0].set_ylabel('Provisional phase phi')
    axes[0].set_title('Output representation')
    phase=np.linspace(-1.2,1.2,601)
    allowed=np.abs(phase)<1
    axes[1].plot(phase,np.ones_like(phase),label='Linear')
    axes[1].plot(phase[allowed],1-phase[allowed]**2,label='tanh')
    axes[1].plot(phase,1/(1+3*phase**2),label='Inverse cubic')
    axes[1].scatter([-1,1],[.25,.25],color='C2',s=24,zorder=4)
    axes[1].set_xlabel('Provisional physical phase');axes[1].set_ylabel('Output sensitivity d(phi)/d(z)')
    axes[1].set_title('Finite sensitivity at pure phases')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.savefig(output/'twophase-map-conservation.png',dpi=180)
    plt.close(fig)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
