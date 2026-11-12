"""Audit complete supports and quadrature of rolling spatial candidates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from ml_collections import ConfigDict

from examples.channel_nsch2d.spatial_sampling import (
    ROW_WEIGHTS, compact_rows, draw_rows, make_tiles, stratified_partition,
    draw_stratified_tiles)

SOURCE = Path('/root/autodl-tmp/tf-logs/channel_nsch2d/20261103-pro6000-integral-random-walls-online-width192-b8-time18-nt36-cold20k-lr1e5-04')


def summary(values):
    return dict(min=float(np.min(values)), median=float(np.median(values)), max=float(np.max(values)))


def audit(output):
    c = ConfigDict(json.loads((SOURCE/'config.json').read_text()))
    c.sample.spatial = ConfigDict(dict(mode='scatter', core_count=512, node_capacity=15633,
        pools=8, refresh_every=50, workers=2))
    settings = [('scatter512', 'scatter', 512, 0, 0), ('tiles512', 'tiles', 512, 0, 0),
                ('strat8_bulk512', 'stratified_tiles', 512, 8, 0),
                ('strat8_wall512', 'stratified_tiles', 512, 8, 2),
                ('strat4_bulk512', 'stratified_tiles', 512, 4, 0),
                ('strat4_wall512', 'stratified_tiles', 512, 4, 8),
                ('strat4_wall1024', 'stratified_tiles', 1024, 4, 16)]
    results = {name: dict(halo=[], unique_core=[], quadrature_sum=[], proxy_error=[]) for name, *_ in settings}
    scalar_checks = 0
    for filename in ('known_inputs.npz', 'known_inputs_final.npz'):
        with np.load(SOURCE/filename, allow_pickle=False) as saved:
            bank = {key: saved[key] for key in saved.files}
        for slot in range(len(bank['points'])):
            data = {key: value[slot] for key, value in bank.items()}
            points = data['points']
            measure = data['transport_measure'].astype(np.float32).astype(float)*data['interior']
            # Prescribed smooth test fields only: no reference-based adaptation.
            proxy = np.exp(-((points[:, 0]-.5)**2+(points[:, 1]-.5)**2)/.04)
            truth = measure@proxy/measure.sum()
            for name, mode, cores, width, wall in settings:
                c.sample.spatial.mode, c.sample.spatial.core_count = mode, cores
                partition = None
                if mode == 'stratified_tiles':
                    c.sample.spatial.tile_width = width
                    c.sample.spatial.wall_groups = wall
                    c.sample.spatial.wall_band = c.nl.delta
                    partition = stratified_partition(data, c)
                rng = np.random.default_rng(np.random.SeedSequence([20261112, slot, cores, width, wall]))
                for draw in range(16):
                    rows, weights = (draw_stratified_tiles(data, c, rng, partition)
                        if partition is not None else draw_rows(data, c, rng, tiles=make_tiles(c.sample.Nx, c.sample.Ny)))
                    compact, halo = compact_rows(data, rows, weights, 15633)
                    record = results[name]
                    record['halo'].append(halo)
                    record['unique_core'].append(len(np.unique(rows)))
                    record['quadrature_sum'].append(weights.sum())
                    record['proxy_error'].append(weights@proxy[rows]-truth)
                    if draw == 0:
                        field = np.sin(points[:, 0])*np.cos(2*points[:, 1])
                        compact_field = np.sin(compact['points'][:, 0])*np.cos(2*compact['points'][:, 1])
                        for key in ROW_WEIGHTS:
                            full = np.sum(data[key][rows]*(field[data['neighbors'][rows]]-field[rows, None]), axis=-1)
                            sampled = np.sum(compact[key]*(compact_field[compact['neighbors']]
                                -compact_field[compact['core_indices'], None]), axis=-1)
                            np.testing.assert_array_equal(full, sampled)
                            scalar_checks += 1
                        for key in ('boundary_gradient_x', 'boundary_gradient_y'):
                            np.testing.assert_array_equal(data[key], compact[key])
                            active = data[key] != 0
                            np.testing.assert_array_equal(field[data['boundary_neighbors']][active],
                                compact_field[compact['boundary_neighbors']][active])
                        for key in ('condition', 'condition_weights', 'boundary', 'boundary_weights'):
                            if key in data:
                                np.testing.assert_array_equal(compact[key], data[key])
    result = dict(status='passed', source=str(SOURCE), timestamp=datetime.now(timezone.utc).isoformat(),
        geometries=32, draws_per_geometry=16, exact_operator_row_checks=scalar_checks,
        field_scope='Synthetic spatial quadrature probe; not a learned solution error',
        modes={name: {key: summary(value) if key != 'proxy_error' else
                      dict(bias=float(np.mean(value)), rmse=float(np.sqrt(np.mean(np.square(value)))))
                      for key, value in record.items()} for name, record in results.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)
