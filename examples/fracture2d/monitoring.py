"""TensorBoard diagnostics and native P1 fields at completed loads."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
from tensorboardX import SummaryWriter


class Monitor:
    def __init__(self, root, c, mechanics, label):
        self.out = Path(root)/'figures'
        self.out.mkdir(exist_ok=True)
        self.c, self.label = c, label
        self.tri = Triangulation(*mechanics.coordinates[:mechanics.n_material].T,
                                 mechanics.triangles[:mechanics.n_material_cells])
        self.n = mechanics.n_material
        self.volumes = mechanics.volumes
        self.reference_root, self.reference_curve = None, None
        if c.output.reference_root:
            self.reference_root = Path(c.output.reference_root)
            actual = json.loads((self.reference_root/'identity.json').read_text())['geometry']
            expected = json.loads((Path(c.mesh.prepared_dir)/'identity.json').read_text())
            settings = ('E', 'nu', 'k_res', 'delta_d', 'visibility_tolerance', 'grip_extent', 'h_grip')
            if any(actual['config'][key] != expected['config'][key] for key in settings):
                raise ValueError('monitor reference geometry/model identity differs')
            self.reference_curve = json.loads((self.reference_root/'result.json').read_text())['curve']
        event_dir = Path(c.output.event_dir) if c.output.event_dir else Path(root)/'events'
        self.writer = SummaryWriter(str(event_dir))
        self.writer.add_text('config', '<pre>'+c.to_json(indent=2)+'</pre>', 0)
        self.writer.add_scalar('train/load', 0., 0)
        self.writer.add_scalar('train/reaction', 0., 0)

    def stage(self, row, step):
        for name in ('full_loss', 'inner_AT2', 'damage_rms',
                     'displacement_rms_over_load', 'damage_min', 'damage_max'):
            self.writer.add_scalar('diag/'+name, row[name], step)
        if 'replay_snapshots' in row:
            for name in ('replay_snapshots', 'replay_training_loss', 'replay_pre_update_loss'):
                self.writer.add_scalar('diag/'+name, row[name], step)
        if 'learning_rate' in row:
            self.writer.add_scalar('diag/learning_rate', row['learning_rate'], step)
        if 'damage_tolerance' in row:
            self.writer.add_scalar('diag/damage_tolerance', row['damage_tolerance'], step)
        if 'maximum_healing' in row:
            for name in ('energy_loss', 'irreversibility_loss', 'penalized_loss',
                         'maximum_healing', 'healing_volume_rms', 'healing_volume_fraction'):
                self.writer.add_scalar('irreversibility/'+name, row[name], step)
        self.writer.add_scalar('train/load', row['load'], step)
        self.writer.add_scalar('diag/stage_reaction_for_previous_damage', row['reaction'], step)

    def endpoint(self, row, fields, curve):
        index = row['index']
        for name in ('reaction', 'load', 'damage_max', 'physical_AT2', 'field_stable'):
            self.writer.add_scalar('train/'+name, row[name], index)
        if 'replay_snapshots' in row:
            self.writer.add_scalar('replay/snapshots_at_endpoint', row['replay_snapshots'], index)
        self.writer.add_scalar('train/outer_iterations', row['sweep'], index)
        if 'maximum_healing' in row:
            self.writer.add_scalar('train/maximum_healing', row['maximum_healing'], index)
        if 'learning_rate' in row:
            self.writer.add_scalar('train/learning_rate', row['learning_rate'], index)
        if index % self.c.output.figure_every:
            self.writer.flush()
            return
        q = fields['q'].reshape(-1, 2)[:self.n]
        reference = None
        if self.reference_curve is not None:
            matching = [r for r in self.reference_curve if abs(r['load']-row['load']) < 1e-14]
            if len(matching) != 1:
                raise ValueError('reference must contain exactly one same-load endpoint')
            ref = matching[0]
            path = self.reference_root/f"state_{ref['index']:04d}"/'fields.npz'
            with np.load(path, allow_pickle=False) as saved:
                reference = saved['d'].copy()
            if reference.shape != fields['d'].shape or not np.isfinite(reference).all():
                raise ValueError('reference damage shape/values differ')
            denominator = self.volumes @ reference**2
            if denominator <= 0:
                raise ValueError('relative damage comparison requires nonzero reference damage')
            error = np.sqrt(self.volumes @ (fields['d']-reference)**2/denominator)
            tag = 'direct_fem' if self.c.damage.discretization == 'fem' else 'direct_nonlocal'
            self.writer.add_scalar('eval/'+tag+'/rel_L2_d', float(error), index)
            self.writer.add_scalar('eval/'+tag+'/reaction', ref['reaction'], index)
        for name, values, cmap in [('damage', fields['d'], 'inferno'),
                ('ux', q[:, 0], 'coolwarm'), ('uy', q[:, 1], 'coolwarm'),
                ('history', fields['H'], 'viridis')]:
            limits = (0., 1.) if name == 'damage' else (float(values.min()), float(values.max()))
            datasets = [('Reference', reference), (self.label, values)] if name == 'damage' and reference is not None else [(self.label, values)]
            fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5),
                                     dpi=self.c.output.figure_dpi, squeeze=False)
            for ax, (label, data) in zip(axes.ravel(), datasets):
                image = ax.tripcolor(self.tri, data, shading='gouraud', cmap=cmap,
                                    vmin=limits[0], vmax=limits[1], rasterized=True)
                ax.set_aspect('equal')
                ax.set(xlabel='x', ylabel='y', title=f'{label} {name}, U={row["load"]:.4f}\n'
                       f'range [{limits[0]:.6g}, {limits[1]:.6g}]')
                fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(self.out/f'{name}_{index:04d}.png')
            self.writer.add_figure('train/'+name+'_field', fig, index, close=True)
        fig, ax = plt.subplots(figsize=(7, 4), dpi=self.c.output.figure_dpi)
        ax.plot([0]+[r['load'] for r in curve], [0]+[r['reaction'] for r in curve], '.-', label=self.label)
        if self.reference_curve is not None:
            refs = [r for r in self.reference_curve if r['load'] <= row['load']+1e-14]
            ax.plot([0]+[r['load'] for r in refs], [0]+[r['reaction'] for r in refs], '-', label='Reference')
        ax.legend()
        ax.set(xlabel='Applied displacement', ylabel='Reaction', title=self.label)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(self.out/f'reaction_{index:04d}.png')
        self.writer.add_figure('train/load_reaction_curve', fig, index, close=True)
        self.writer.flush()

    def close(self):
        self.writer.close()
