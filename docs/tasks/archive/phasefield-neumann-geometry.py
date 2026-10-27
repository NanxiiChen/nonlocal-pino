from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
out=Path('/root/nonlocal-pinn/docs/tasks/archive')
fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained')
ax.add_patch(Ellipse((0,0),2.5,1.55,facecolor='#d7eaf5',edgecolor='#25475b',linewidth=2.5))
for xy,r in [((-.43,.16),.21),((.46,-.15),.15)]:
 ax.add_patch(Circle(xy,r,facecolor='white',edgecolor='#25475b',linewidth=2.2,hatch='///'))
ax.text(0,.48,'Binary mixture: domain $\\Omega$',ha='center',fontsize=14)
ax.text(0,-.52,'Phase separation and coarsening',ha='center',fontsize=12,color='#25475b')
ax.annotate('Impermeable holes',xy=(-.43,.16),xytext=(-1.52,1.02),arrowprops=dict(arrowstyle='->'),fontsize=12)
ax.annotate('Closed outer wall',xy=(1.18,.23),xytext=(.56,1.02),arrowprops=dict(arrowstyle='->'),fontsize=12)
ax.text(0,-1.03,r'All walls: $\partial_n\phi=0$, $\partial_n\mu=0$ (local target)',ha='center',fontsize=12)
ax.set(xlim=(-1.6,1.6),ylim=(-1.16,1.18),aspect='equal',title='Proposed first geometry — schematic only')
ax.axis('off')
fig.savefig(out/'phasefield-neumann-geometry.png',dpi=180)
plt.close(fig)
