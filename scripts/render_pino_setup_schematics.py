"""Render portable known-input setup SVGs; no inference or PDE/reference solves."""
import argparse
import json
from pathlib import Path
import sys
import textwrap
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from ml_collections import ConfigDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.icemelting3d.ellipsoids import rotation, validate_parameters
from examples.confined_ch2d.operator_problem import fixed_initial, valid_holes
from examples.channel_nsch2d.geometry import profiles, validate_parameters as validate_walls

NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
BLUE, ORANGE, INK, GRAY = '#2582b8', '#df963c', '#26323e', '#737d89'
WIDTH, PANEL = 1024, 296


def node(parent, tag, **attributes):
    return ET.SubElement(parent, '{'+NS+'}'+tag, {k.replace('_', '-'): str(v) for k, v in attributes.items()})


def text(parent, x, y, value, size):
    element = node(parent, 'text', x=x, y=y, fill=INK, font_size=size)
    element.text = value


def path_data(points, point, close):
    commands = []
    for i, p in enumerate(points):
        x, y = point(p)
        commands.append(('M' if i == 0 else 'L')+f'{x:.3f},{y:.3f}')
    return ' '.join(commands)+(' Z' if close else '')


def line(parent, points, point, color, width):
    node(parent, 'path', d=path_data(points, point, False), fill='none', stroke=color,
         stroke_width=width, stroke_linecap='round', stroke_linejoin='round')


def solid(parent, points, point, opacity):
    node(parent, 'path', d=path_data(points, point, True), fill=BLUE, fill_opacity=opacity)


def ellipse_projection(case, group, bx, by, index):
    p = np.asarray(case['parameters'][index])
    validate_parameters(p)
    initial = case['initial']
    if (np.any(p[:3] < initial['center_min']) or np.any(p[:3] > initial['center_max'])
            or np.any(p[3:6] < initial['axis_min']) or np.any(p[3:6] > initial['axis_max'])):
        raise ValueError('Illustrative ellipsoid lies outside the documented prior')
    view = np.array([1.7, -2.4, 1.7]); view /= np.linalg.norm(view)
    right = np.cross([0., 0., 1.], view); right /= np.linalg.norm(right)
    up = np.cross(view, right)
    projection = np.stack((right, -up))
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    projected = cube@projection.T
    low = projected.min(0); extent = np.ptp(projected, axis=0); scale = extent.max()
    def screen(points):
        return (points@projection.T-low)/scale+(1-extent/scale)/2
    point = lambda q: (bx+13+270*q[0], by+270*q[1])
    edges = [[i, j] for i in range(8) for j in range(i+1, 8) if np.sum(cube[i] != cube[j]) == 1]
    for edge in edges:
        line(group, screen(cube[edge]), point, GRAY, 1.2)
    r = rotation(p[6:], np)
    theta = np.linspace(0, 2*np.pi, 100)
    tt, ll = np.meshgrid(theta, np.linspace(-np.pi/2, np.pi/2, 49))
    sphere = np.stack((np.cos(ll)*np.cos(tt), np.cos(ll)*np.sin(tt), np.sin(ll)), axis=-1).reshape(-1, 3)
    surface = (sphere*p[3:6])@r.T+p[:3]
    hull = ConvexHull(surface@projection.T)
    outline = screen(surface[hull.vertices])
    solid(group, outline, point, .35)
    for normal in range(3):
        circle = np.zeros((len(theta), 3)); dims = [j for j in range(3) if j != normal]
        circle[:, dims[0]] = np.cos(theta); circle[:, dims[1]] = np.sin(theta)
        line(group, screen((circle*p[3:6])@r.T+p[:3]), point, BLUE, 1)
    line(group, np.vstack((outline, outline[0])), point, ORANGE, 2.2)


def electro(case, group, bx, by, index):
    coefficients = np.asarray(case['coefficients'][index])
    prior = case['prior']
    if coefficients.shape != (prior['n_modes'], 3) or not np.isfinite(coefficients).all():
        raise ValueError('Invalid prescribed rough-interface coefficients')
    if (np.any(coefficients[:, 0] < .3*prior['amplitude'])
            or np.any(coefficients[:, 0] > prior['amplitude'])
            or np.any(coefficients[:, 1] < prior['freq_lo'])
            or np.any(coefficients[:, 1] > prior['freq_hi'])
            or np.any(coefficients[:, 2] < 0) or np.any(coefficients[:, 2] > 2*np.pi)):
        raise ValueError('Illustrative rough profile lies outside the documented prior')
    x = np.linspace(0, 1, 2000)
    angle = 2*np.pi*x[:, None]*coefficients[:, 1]+coefficients[:, 2]
    y = prior['y0']+np.sum(coefficients[:, 0]*(np.sin(angle)+np.cos(angle+.3)), axis=-1)
    low, high = case['height_acceptance']
    if not low < y.min() <= y.max() < high:
        raise ValueError('Illustrative rough profile violates height acceptance')
    point = lambda q: (bx+PANEL*q[0], by+PANEL*(.5-q[1]))
    ids = np.r_[np.arange(0, len(x), 8), len(x)-1]
    curve = np.column_stack((x[ids], y[ids]))
    solid(group, [[0, 0], [1, 0], [1, .5], [0, .5]], point, .08)
    solid(group, np.vstack(([0, 0], curve, [1, 0])), point, .35)
    line(group, [[0, 0], [1, 0], [1, .5], [0, .5], [0, 0]], point, GRAY, 1.2)
    line(group, curve, point, ORANGE, 2.2)


def spinodal_texture(case, group):
    """One shared ambient initial texture, reused by all geometry samples."""
    c = ConfigDict(dict(geometry=case['geometry'], initial=case['initial']))
    point = lambda q: (PANEL*q[0], PANEL*(.9-q[1]))
    defs = node(group, 'defs')
    field = node(defs, 'g', id='spinodal-shared-texture')
    node(field, 'rect', x=0, y=0, width=PANEL, height=PANEL*.8, fill=BLUE, fill_opacity=.08)
    x, y = np.linspace(0, 1, 241), np.linspace(.1, .9, 193)
    xx, yy = np.meshgrid(x, y)
    values = fixed_initial(np.stack((xx, yy), axis=-1).reshape(-1, 2), c, xp=np).reshape(xx.shape)
    fig, ax = plt.subplots()
    contour = ax.contourf(xx, yy, values, levels=[float(values.min())-1e-7, 0, float(values.max())+1e-7])
    for vertices, codes in zip(contour.allsegs[1], contour.allkinds[1], strict=True):
        commands = []
        for p, code in zip(vertices, codes, strict=True):
            xxp, yyp = point(p)
            commands.append('Z' if code == 79 else ('M' if code == 1 else 'L')+f'{xxp:.3f},{yyp:.3f}')
        node(field, 'path', d=' '.join(commands), fill=BLUE, fill_opacity=.35, fill_rule='evenodd')
    plt.close(fig)


def spinodal(case, group, bx, by, index):
    c = ConfigDict(dict(geometry=case['geometry'], initial=case['initial'], operator=case['prior']))
    holes = np.asarray(case['holes'][index])
    if (tuple(c.geometry.center) != (.5, .5) or tuple(c.geometry.axes) != (.5, .4)
            or not valid_holes(holes, c)
            or np.any(holes[:, :2] < np.asarray(c.operator.center_min))
            or np.any(holes[:, :2] > np.asarray(c.operator.center_max))
            or np.any(holes[:, 2] < np.asarray(c.operator.radius_min))
            or np.any(holes[:, 2] > np.asarray(c.operator.radius_max))):
        raise ValueError('Illustrative hole geometry is not admissible')
    point = lambda q: (bx+PANEL*q[0], by+PANEL*(.9-q[1]))
    cx, cy = point([.5, .5]); rx, ry = PANEL*.5, PANEL*.4
    domain = f'M{cx-rx},{cy} A{rx},{ry} 0 1 0 {cx+rx},{cy} A{rx},{ry} 0 1 0 {cx-rx},{cy} Z '
    for x, y, radius in holes:
        xx, yy = point([x, y]); rr = radius*PANEL
        domain += f'M{xx-rr},{yy} A{rr},{rr} 0 1 0 {xx+rr},{yy} A{rr},{rr} 0 1 0 {xx-rr},{yy} Z '
    defs = node(group, 'defs'); clip = node(defs, 'clipPath', id='spinodal-domain-'+str(index))
    node(clip, 'path', d=domain, clip_rule='evenodd', fill_rule='evenodd')
    field = node(group, 'g', clip_path='url(#spinodal-domain-'+str(index)+')')
    node(field, 'use', href='#spinodal-shared-texture', transform=f'translate({bx},{by})')
    node(group, 'ellipse', cx=cx, cy=cy, rx=rx, ry=ry, fill='none', stroke=GRAY, stroke_width=1.2)
    for x, y, radius in holes:
        xx, yy = point([x, y])
        node(group, 'circle', cx=xx, cy=yy, r=radius*PANEL, fill='none', stroke=ORANGE, stroke_width=2.2)


def channel(case, group, bx, by, index):
    c = ConfigDict(dict(geometry=case['geometry']))
    if (c.geometry.length, c.geometry.height) != (4., 1.):
        raise ValueError('Setup schematic requires the documented4x1channel')
    parameters = np.asarray(case['parameters'][index])
    validate_walls(parameters, c)
    x = np.linspace(0, c.geometry.length, 321)
    lower, upper = profiles(x, parameters, c, xp=np)
    point = lambda q: (bx+20+(PANEL-40)*q[0]/4, by+(PANEL-40)*(1-q[1])/4)
    lower, upper = np.column_stack((x, lower)), np.column_stack((x, upper))
    solid(group, np.vstack((lower, upper[::-1])), point, .08)
    for curve in (lower, upper):
        line(group, curve, point, GRAY, 1.2)
        inside = (curve[:, 0] >= c.geometry.start) & (curve[:, 0] <= c.geometry.end)
        line(group, curve[inside], point, ORANGE, 2.2)
    for xx in (0, 4):
        line(group, [[xx, 0], [xx, 1]], point, GRAY, 1.2)
    xx, yy = point(case['initial']['center'])
    node(group, 'circle', cx=xx, cy=yy, r=case['initial']['radius']*(PANEL-40)/4,
         fill=BLUE, fill_opacity=.35)
    for xx in (bx+1, bx+PANEL-17):
        ymid = by+(PANEL-40)/8
        line(group, [[xx, ymid], [xx+15, ymid]], lambda p: p, INK, 1.4)
        line(group, [[xx+11, ymid-3], [xx+15, ymid], [xx+11, ymid+3]], lambda p: p, INK, 1.4)


SPECS = {
    'icemelting3d': dict(title='Ice melting 3D | random initial condition',
        varying='Initial ellipsoid center, three semiaxes and 3D orientation',
        fixed='Cube, material parameters and melting drive; time 0-5 s',
        phase='Ice; the surrounding phase is water', variable='Random initial ice interface', body=270,
        footer=['BC: Closed interactions; no exchange with the exterior of the cube.',
                'Physics: Driven Allen-Cahn melting -> phase field phi(x,y,z,t).',
                'Condition: Initial phase volume -> 3D patch tokens -> shared space-time queries.',
                'Setup only. Same orthographic view; no predicted evolution or temperature field.']),
    'electropolish2d': dict(title='Electropolish 2D | random initial condition',
        varying='Smooth initial metal/electrolyte interface (six random modes)',
        fixed='Rectangle [0,1] x [0,0.5], material and boundary driving; time 0-5 s',
        phase='Metal; electrolyte lies above', variable='Random initial interface', body=148,
        footer=['BC: Top electrolyte (phi,c)=(0,0); bottom metal (1,1); closed side walls.',
                'Physics: Allen-Cahn + concentration transport / KKS -> phi(x,y,t), c(x,y,t).',
                'Condition: Initial phase image -> 2D patch tokens -> shared space-time queries.',
                'Setup only. A sharp outline represents the location of the actual diffuse interface.']),
    'confined_ch2d': dict(title='Spinodal decomposition 2D | random geometry',
        varying='Positions and radii of two impermeable circular holes',
        fixed='Outer ellipse, the same ten-mode initial field and materials; model time 0-3',
        phase='Positive part of the fixed initial fluctuation', variable='Random hole boundary', body=236.8,
        footer=['BC: Neutral wetting and zero flux on outer/hole walls; no interactions through holes.',
                'Physics: Cahn-Hilliard phase separation/coarsening -> phi; mu follows its constitutive law.',
                'Condition: Boundary + domain/initial-field points -> point tokens -> queries in each domain.',
                'Setup only. Exactly the same ambient initial texture is restricted to each geometry.']),
    'channel_nsch2d': dict(title='Two-phase flow 2D | random channel geometry',
        varying='Independent smooth upper/lower wall constrictions (eight coefficients per wall)',
        fixed='Initial droplet, inlet driving and materials; channel 4 x 1; model time 0-6',
        phase='Fixed initial droplet', variable='Random wall segment; straight ends stay fixed', body=64,
        footer=['BC: Left prescribed carrier/velocity with a fixed ramp; right zero hydrodynamic traction.',
                'Walls: No slip, neutral wetting and zero chemical flux; minimum aperture [0.55,0.70].',
                'Physics: Full Navier-Stokes + Cahn-Hilliard coupling -> phi, mu, velocity and pressure.',
                'Condition: Boundary + domain/initial points -> point tokens -> queries in each domain.',
                'Setup only. Droplets are identical; arrows indicate driving direction, not initial velocity.']),
}


def render(case, record, output):
    spec = SPECS[case]
    root = ET.Element('{'+NS+'}svg', {'width': str(WIDTH), 'role': 'img', 'aria-labelledby': 'title description'})
    node(root, 'title', id='title').text = spec['title']
    node(root, 'desc', id='description').text = 'Known input/setup schematic only; three samples. No predicted or evolved solution fields.'
    background = node(root, 'rect', width=WIDTH, fill='#ffffff')
    group = node(root, 'g', font_family='DejaVu Sans, Arial, sans-serif')
    text(group, 28, 34, spec['title'], 24)
    text(group, 28, 64, 'Varying: '+spec['varying'], 17)
    text(group, 28, 88, 'Fixed: '+spec['fixed'], 16)
    draw = {'icemelting3d': ellipse_projection, 'electropolish2d': electro,
            'confined_ch2d': spinodal, 'channel_nsch2d': channel}[case]
    if case == 'confined_ch2d':
        spinodal_texture(record, group)
    for index in range(3):
        bx = 28+336*index
        text(group, bx, 128, 'Sample '+chr(65+index), 17)
        draw(record, group, bx, 148, index)
    y = 148+spec['body']+28
    node(group, 'rect', x=28, y=y-12, width=16, height=11, fill=BLUE, fill_opacity=.45)
    text(group, 52, y, spec['phase'], 15.5)
    line(group, [[560, y-6], [582, y-6]], lambda p: p, ORANGE, 2.2)
    text(group, 592, y, spec['variable'], 15.5)
    y += 28
    for value in spec['footer']:
        for part in textwrap.wrap(value, width=112):
            text(group, 28, y, part, 16)
            y += 23
    height = y+8
    root.set('height', str(height)); root.set('viewBox', f'0 0 {WIDTH} {height}')
    background.set('height', str(height))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)
    return dict(path=str(output), bytes=output.stat().st_size, samples=3, known_inputs_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'artifacts/pino_setup_schematics.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['schema'] != 1 or manifest['pde_solves'] != 0 or manifest['inference_runs'] != 0:
        raise ValueError('Known-input-only setup manifest required')
    results = []
    for case, record in manifest['cases'].items():
        results.append(render(case, record, ROOT/record['output']))
    if sum(r['bytes'] for r in results) >= 256*1024:
        raise ValueError('Four setup SVGs must total less than256KiB')
    print(json.dumps(dict(status='rendered_known_input_setups', figures=results,
                          total_bytes=sum(r['bytes'] for r in results), pde_solves=0, inference_runs=0), indent=2))


if __name__ == '__main__':
    main()
