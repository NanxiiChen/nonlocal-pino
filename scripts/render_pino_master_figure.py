"""Draw the source-audited PINO architecture/training schematic as editable art."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np

from scripts.render_pino_derivative_comparisons import publication_style

ROOT = Path(__file__).resolve().parents[1]


def tint(color, fraction=.10):
    return tuple(1-fraction*(1-v) for v in to_rgb(color))


class Drawing:
    def __init__(self, ax, palette):
        self.ax, self.colors, self.texts = ax, palette, []

    def text(self, x, y, value, size=6, ha='left', **kwargs):
        # Helvetica lacks these two symbols; use the explicitly declared sans math face.
        value=value.replace('→',r'$\rightarrow$').replace('⊙',r'$\odot$')
        item = self.ax.text(x, y, value, fontsize=size, ha=ha, va='center',
                            color=self.colors['ink'], linespacing=1.16, zorder=10, **kwargs)
        self.texts.append(item)
        return item

    def box(self, x, y, w, h, label='', color='blue', size=6, dashed=False, fill=True):
        c = self.colors[color]
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.2',
            linewidth=.65,edgecolor=c,facecolor=tint(c,.09) if fill else 'white',
            linestyle=(0,(4,3)) if dashed else '-',zorder=1))
        if label:
            self.text(x+w/2,y+h/2,label,size,ha='center')

    def panel(self,x,y,w,h,title,color):
        self.box(x,y,w,h,color=color,dashed=True,fill=False)
        self.text(x+2.5,y+3.3,title,7)

    def line(self, points, color='gray', width=.65, dashed=False, zorder=2):
        self.ax.plot(*np.asarray(points).T, color=self.colors[color], lw=width,
                     ls=(0,(3,2)) if dashed else '-',zorder=zorder)

    def arrow(self, points, color='gray', width=.65, dashed=False):
        vertices=np.asarray(points,dtype=float)
        path=MplPath(vertices,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        self.ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=5,
            linewidth=width,color=self.colors[color],linestyle=(0,(3,2)) if dashed else '-',
            zorder=3))

    def tokens(self,x,y,n,color,vertical=False):
        for i in range(n):
            dx,dy=(0,2.3*i) if vertical else (2.3*i,0)
            self.ax.add_patch(Rectangle((x+dx,y+dy),1.7,2,
                facecolor=tint(self.colors[color],.35),edgecolor=self.colors[color],lw=.45,zorder=4))

    def geometry(self,x,y,w,h,variant=0,points=False):
        self.ax.add_patch(Rectangle((x,y),w,h,facecolor='#f7f8f9',edgecolor='#777777',lw=.45,zorder=2))
        holes=[(.28+.06*variant,.34,.16),(.69-.04*variant,.65,.17-.015*variant)]
        if points:
            for yy in np.linspace(.09,.91,9):
                for xx in np.linspace(.08,.92,10):
                    if all(((xx-cx)*w)**2+((yy-cy)*h)**2>(r*min(w,h))**2 for cx,cy,r in holes):
                        self.ax.add_patch(Circle((x+xx*w,y+yy*h),.19,color=self.colors['green'],lw=0,zorder=3))
        for cx,cy,r in holes:
            self.ax.add_patch(Circle((x+cx*w,y+cy*h),r*min(w,h),facecolor='white',edgecolor=self.colors['blue'],lw=.55,zorder=4))
            if points:
                for a in np.linspace(0,2*np.pi,14,endpoint=False):
                    self.ax.add_patch(Circle((x+cx*w+r*min(w,h)*np.cos(a),y+cy*h+r*min(w,h)*np.sin(a)),.22,
                        color=self.colors['blue'],lw=0,zorder=5))

    def field(self,x,y,w,h,patch=False,evolved=False):
        xx,yy=np.meshgrid(np.linspace(0,1,72),np.linspace(0,1,64))
        level=(.38+.06*np.sin(2*np.pi*xx)+.035*np.cos(4*np.pi*xx) if evolved
               else .50+.10*np.sin(2*np.pi*xx)+.07*np.cos(4*np.pi*xx))
        field=np.tanh((yy-level)/.06)
        self.ax.imshow(field,extent=(x,x+w,y+h,y),origin='upper',cmap='RdBu_r',vmin=-1,vmax=1,
                       interpolation='bilinear',zorder=2,aspect='auto')
        self.ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#777777',lw=.5,zorder=4))
        if patch:
            for frac in (.25,.5,.75):
                self.line([(x+frac*w,y),(x+frac*w,y+h)],width=.5,zorder=4)
                self.line([(x,y+frac*h),(x+w,y+frac*h)],width=.5,zorder=4)

    def core_halo(self,x,y,w,h):
        # Every visible halo point is derived from the displayed core stencil.
        xs,ys=np.meshgrid(np.arange(11),np.arange(7))
        p=np.c_[xs.ravel(),ys.ravel()].astype(float)
        cores=np.array([16,37,52,60]);radius=1.65
        halo=(np.linalg.norm(p[:,None]-p[cores][None],axis=-1)<=radius).any(axis=1)
        q=np.c_[x+w*p[:,0]/10,y+h*p[:,1]/6]
        for i,point in enumerate(q):
            color='#d9dcdf';r=.20
            if halo[i]:color=self.colors['cyan'];r=.32
            if i in cores:color=self.colors['orange'];r=.52
            self.ax.add_patch(Circle(point,r,facecolor=color,edgecolor='white',lw=.15,zorder=4))
        for i in cores:
            self.ax.add_patch(Circle(q[i],radius*w/10,fill=False,ec=self.colors['cyan'],lw=.35,
                                    ls=(0,(2,2)),zorder=3))


def compose(d):
    # (a) The shared data-preparation pipeline; no target trajectories are built.
    d.panel(2,2,166,28,'(a) Parameterized inputs and rolling sampling','purple')
    d.field(5,10,10,10)
    d.geometry(18,10,10,10,0)
    d.geometry(30,10,10,10,1)
    d.text(22.5,24,'Initial field or geometry',5.8,ha='center')
    d.arrow([(41,15),(47,15)])
    d.box(48,10,30,10,'Mesh / quadrature\nNonlocal operators','purple',5.8)
    d.text(63,24,'Background CPU workers',5.7,ha='center')
    d.arrow([(79,15),(85,15)])
    for dx,dy in [(2,-2),(1,-1),(0,0)]:
        d.box(88+dx,11+dy,25,9,color='purple')
    d.text(100.5,15.5,'Rolling input pool',5.8,ha='center')
    d.arrow([(111,21),(111,24),(91,24),(91,21)],color='purple',dashed=True)
    d.text(101,27,'Replace / reuse',5.5,ha='center')
    d.arrow([(115,15),(123,15)])
    d.core_halo(127,10,23,12)
    d.text(157,11,'Core',5.5)
    d.text(157,16,'Halo',5.5)
    d.ax.add_patch(Circle((154.8,11),.65,color=d.colors['orange'],lw=0))
    d.ax.add_patch(Circle((154.8,16),.5,color=d.colors['cyan'],lw=0))
    d.text(145,26,'Random core + full halo',5.6,ha='center')
    d.text(85,32.3,'Known inputs only; no reference trajectories',5.5,ha='center')
    d.arrow([(34,30.4),(34,35.5)])
    d.arrow([(152,30.4),(152,35.5)])

    # (b) Fixed geometry: patch projection, positional code, self-attention.
    d.panel(2,36,65,44,'(b) Fixed domain: patch tokens','blue')
    d.field(6,45,12,12,patch=True)
    d.text(12,60,'2D / 3D',5.5,ha='center')
    d.arrow([(18.8,50),(22,50)])
    d.box(22.5,45,16,12,'Patch lift\n+ position','blue',5.7)
    d.arrow([(39.2,50),(43,50)])
    d.box(43.5,45,19,12,'Self-\nattention','blue',6)
    d.arrow([(53,57.3),(53,65)],color='blue')
    d.text(55,61.2,'K, V',5.5)
    d.box(6,67,22,9,'x → Fourier','green',6)
    d.arrow([(28.5,71.5),(37,71.5)],color='green')
    d.text(32,69,'Q',5.5,ha='center')
    d.box(37.5,65.5,25,12,'Spatial cross-\nattention','blue',6)

    # (c) Geometry point lifts and two separately normalized context reads.
    d.panel(72,36,96,44,'(c) Parameterized geometry: point tokens','green')
    d.geometry(75,45,12,24,0,points=True)
    d.text(90,44,'Boundary: x, n, type',5.6)
    d.text(90,59,'Domain: x, initial field',5.6)
    d.tokens(90,48,5,'blue')
    d.tokens(90,63,5,'green')
    d.arrow([(102,49.5),(105.5,49.5)],color='blue')
    d.arrow([(102,64.5),(105.5,64.5)],color='green')
    d.box(106,46,12,8,'MLP','blue')
    d.box(106,61,12,8,'MLP','green')
    d.arrow([(118.5,50),(131.5,50)],color='blue')
    d.arrow([(118.5,65),(131.5,65)],color='green')
    d.text(125,47,'K, V; ds',5.5,ha='center')
    d.text(125,62,'K, V; dA',5.5,ha='center')
    d.box(132,45.5,18,9,'Weighted\ncross-attn','blue',5.8)
    d.box(132,60.5,18,9,'Weighted\ncross-attn','green',5.8)
    d.text(141,57.5,'Shared Q',5.5,ha='center')
    d.arrow([(141,56),(141,54.6)])
    d.arrow([(141,59),(141,60.3)])
    d.arrow([(150.5,50),(160,55.3)],color='blue')
    d.arrow([(150.5,65),(160,59.7)],color='green')
    d.ax.add_patch(Circle((160,57.5),2.2,ec=d.colors['green'],fc='white',lw=.65,zorder=4))
    d.text(160,57.5,'avg',5,ha='center')
    d.arrow([(160,60.1),(160,71)])
    d.box(150,71.5,15,6,'Proj. + MLP','green',5.5)
    d.text(91,74,'Position features + masks',5.5)
    d.text(138,77,'× L blocks',5.5,ha='center')

    # One alternative encoder supplies h(x); there is no mixing of architectures.
    d.line([(50,77.8),(50,82),(158,82),(158,77.8)],width=.6)
    d.arrow([(50,82),(16,82),(16,96)])
    d.text(104,82,'Either encoder → spatial features',5.6,ha='center',
           bbox=dict(facecolor='white',edgecolor='none',pad=1))

    # (d) All spatial attention precedes time; each MLP layer has its own FiLM.
    d.panel(2,86,166,27,'','green')
    d.text(34,89,'(d) Temporal decoder',7)
    d.text(120,89,'t → Fourier + MLP',6,ha='center')
    d.line([(120,90.7),(120,93),(46,93)],color='purple')
    for center in [46,73,100]:
        d.arrow([(center,93),(center,96)],color='purple')
    d.box(5,96.5,22,10,'Spatial cache\nreuse across t','green',5.6)
    for x in [35,62,89]:
        d.box(x,96.5,22,10,'Linear → FiLM\n→ GELU','green',5.8)
    for a,b in [(27.5,34.5),(57.5,61.5),(84.5,88.5),(111.5,119.5),(138.5,147.5)]:
        d.arrow([(a,101.5),(b,101.5)])
    d.text(73,110,'(1 + γ) ⊙ z + β   ·   separate γ, β at every layer',5.8,ha='center')
    d.box(120,96.5,18,10,'Head +\noutput map','green',5.8)
    d.field(148,96,16,11,evolved=True)
    d.text(156,92.5,'u(x, t)',6,ha='center')
    d.arrow([(156,107.5),(156,115),(13,115),(13,118)])
    d.text(83,115,'Predictions on core + halo',5.6,ha='center',
           bbox=dict(facecolor='white',edgecolor='none',pad=1))

    # (e-g) A schematic toolbox; none of these boxes claims all options are on.
    d.panel(2,118,65,29,'(e) Nonlocal physics','cyan')
    center=np.array([12,135]);rad=6
    d.ax.add_patch(Circle(center,rad,ec=d.colors['cyan'],fc=tint(d.colors['cyan'],.09),lw=.5,ls=(0,(3,2))))
    for angle in np.linspace(0,2*np.pi,7,endpoint=False):
        pt=center+rad*.8*np.array([np.cos(angle),np.sin(angle)])
        d.line([center,pt],color='cyan',width=.4)
        d.ax.add_patch(Circle(pt,.48,color=d.colors['cyan'],lw=0,zorder=4))
    d.ax.add_patch(Circle(center,.7,color=d.colors['orange'],lw=0,zorder=4))
    d.text(17,128,'δ',6)
    d.text(23,128,'D u = Σ w (u − u′)',6.6)
    d.text(23,133,'Spatial: D ≈ −Δ',6)
    d.text(23,138,'Time: JVP wrt t',6)
    d.text(34.5,144,'Domain-restricted stencil; core residuals',5.5,ha='center')
    d.arrow([(67.5,132),(71,132)])

    d.panel(72,118,44,29,'(f) Physics-only loss','orange')
    d.box(76,126,36,8,'PDE residuals','orange',6)
    d.text(94,137.5,'Scale / precondition',5.8,ha='center')
    d.text(94,143,'Soft IC / BC or hard lifts',5.5,ha='center')
    d.arrow([(116.5,132),(120,132)])

    d.panel(121,118,47,29,'(g) Weighting and update','purple')
    for j,height in enumerate([5,8,6]):
        d.ax.add_patch(Rectangle((125+j*3,136-height),1.7,height,fc=d.colors[['orange','blue','green'][j]],lw=0))
    d.line([(124,136),(134,136)],width=.5)
    d.text(129,139,'r  ic  bc',5,ha='center')
    d.arrow([(136,131),(144,131)])
    d.box(145,126,19,10,'Adam /\nSOAP','purple',6)
    d.text(145,143,'Fixed / gradient-norm weights',5.3,ha='center')
    d.text(85,148.4,'Case-dependent: core/halo, residual transforms and balancing; causal weighting is optional.',5.5,ha='center')


def render(spec):
    for source in spec['source_files']:
        (ROOT/source).resolve(strict=True)
    rc,font=publication_style(spec['typography'])
    rc.update({'svg.fonttype':'none','svg.hashsalt':'pino-master-figure-v1'})
    with plt.rc_context(rc):
        fig=plt.figure(figsize=np.asarray(spec['figure_mm'])/25.4)
        ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,170);ax.set_ylim(150,0);ax.set_axis_off()
        d=Drawing(ax,spec['palette']);compose(d)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        texts=[]
        for text in d.texts:
            b=text.get_window_extent(renderer)
            if not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1):
                raise ValueError('Text outside page: '+text.get_text())
            texts.append(dict(text=text.get_text(),size_pt=text.get_fontsize(),bounds_px=list(b.bounds)))
        for kind,out in spec['outputs'].items():
            path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
            metadata={'CreationDate':None,'ModDate':None} if kind=='pdf' else {'Date':None} if kind=='svg' else None
            fig.savefig(path,dpi=spec['dpi'],metadata=metadata)
        plt.close(fig)
    svg=Path(spec['outputs']['svg'])
    ns='http://www.w3.org/2000/svg'
    tree=ET.parse(svg);root=tree.getroot()
    root.set('role','img');root.set('aria-labelledby','master-title master-description')
    title=ET.Element('{'+ns+'}title',id='master-title')
    title.text='Nonlocal PINO architecture and physics-only training'
    description=ET.Element('{'+ns+'}desc',id='master-description')
    description.text=('Background rolling geometry/operator inputs and core-halo sampling; '
        'alternative patch or two-stream point encoders; separate weighted spatial reads; '
        'three subsequent decoder layers each with time FiLM; nonlocal spatial residuals, '
        'time JVP and case-dependent loss control and optimization. Textures are schematic.')
    root.insert(0,title);root.insert(1,description)
    ET.register_namespace('',ns);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    tree.write(svg,encoding='utf-8',xml_declaration=True)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    sys.path.insert(0,str(Path(spec['pdf_audit_packages']).resolve(strict=True)))
    import fitz
    with fitz.open(spec['outputs']['pdf']) as doc:
        page=doc[0]
        spans=[s for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans']]
        if any(not 4.99<=s['size']<=7.01 for s in spans):raise ValueError('PDF type size outside5–7pt')
        fonts=page.get_fonts(full=True)
        if any(f[2]=='Type3' or not doc.extract_font(f[0])[3] for f in fonts):raise ValueError('Noneditable/unembedded PDF font')
        np.testing.assert_allclose([page.rect.width,page.rect.height],np.asarray(spec['figure_mm'])*72/25.4,atol=.002)
        page.get_pixmap(matrix=fitz.Matrix(2,2)).save(str(Path(spec['outputs']['pdf']).with_name('pdf_preview.png')))
    record=dict(specification=spec,renderer=str(Path(__file__).relative_to(ROOT)),font=font,
        text=texts,pdf_fonts=fonts,font_range_pt=[min(s['size'] for s in spans),max(s['size'] for s in spans)],
        output_bytes={k:Path(v).stat().st_size for k,v in spec['outputs'].items()},
        raster_scope='Only schematic input/output texture images; text, arrows, nodes and all other elements are vector',
        visual_review='pending',numerical_changes=False)
    (ROOT/spec['receipt']).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'outputs':spec['outputs'],'bytes':record['output_bytes']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    render(json.loads(parser.parse_args().spec.read_text()))
