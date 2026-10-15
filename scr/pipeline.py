"""Main controller for the line-emission map pipeline."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    # Also allow: python Pipeline/scr/pipeline.py config.json
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Pipeline.scr import simba
    from Pipeline.scr.emissivity_tables import calculate_particle_line_luminosity
    from Pipeline.scr.make_xxv_cube import make_xxv_cube
else:
    from . import simba
    from .emissivity_tables import calculate_particle_line_luminosity
    from .make_xxv_cube import make_xxv_cube


def run_line_emission_pipeline(snapshot_filename, gas_rows, *, table_type, table_filename, line_label, redshift, unit_factors, center_kpc, x_edges_kpc, y_edges_kpc, velocity_edges_kms, los_axis='z', systemic_velocity_kms=0.0) -> dict:
    """Run the line-emission calculation from CAESAR snapshot rows.

    Returns: result (dict): luminosity cube, bin edges, and run metadata.
    """
    center = np.asarray(center_kpc, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError('center_kpc must contain three finite coordinates.')

    bin_edges = []
    for name, values in (
        ('x_edges_kpc', x_edges_kpc),
        ('y_edges_kpc', y_edges_kpc),
        ('velocity_edges_kms', velocity_edges_kms),
    ):
        edges = np.asarray(values, dtype=np.float64)
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.all(np.isfinite(edges))
            or not np.all(np.diff(edges) > 0)
        ):
            raise ValueError(
                f'{name} must be a finite 1D array with strictly increasing values.'
            )
        bin_edges.append(edges)
    x_edges_kpc, y_edges_kpc, velocity_edges_kms = bin_edges

    redshift = float(redshift)
    if not np.isfinite(redshift):
        raise ValueError('redshift must be finite.')

    gas_fields = simba.load_physical_gas_fields(
        snapshot_filename, gas_rows, unit_factors
    )
    for axis_index, axis in enumerate(('x', 'y', 'z')):
        gas_fields[f'{axis}_kpc'] -= center[axis_index]

    emission = calculate_particle_line_luminosity(
        gas_fields,
        table_type=table_type,
        table_filename=table_filename,
        redshift=redshift,
        line_label=line_label,
    )
    result = make_xxv_cube(
        gas_fields,
        emission['line_luminosity_erg_s'],
        x_edges_kpc,
        y_edges_kpc,
        velocity_edges_kms,
        los_axis=los_axis,
        systemic_velocity_kms=systemic_velocity_kms,
    )
    result.update({
        'input_source': 'simba',
        'table_type': table_type,
        'line_label': line_label,
        'redshift': redshift,
        'selected_particle_count': len(gas_rows),
        'center_kpc': center,
    })
    return result


def main_controller():
    """Run the active row-based pipeline from a JSON configuration.

    Returns: none (NoneType): saves the luminosity cube to an NPZ file.
    """
    parser = argparse.ArgumentParser(description='Run the line-emission cube pipeline from a JSON config.')
    parser.add_argument('config', help='Path to the pipeline JSON configuration')
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    with config_path.open('r', encoding='utf-8') as config_file:
        config = json.load(config_file)

    def resolve_path(value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        return str(path.resolve())

    snapshot_filename = resolve_path(config['snapshot_filename'])
    input_source = config.get('input_source', 'simba').lower()
    if input_source != 'simba':
        raise ValueError(
            f'This controller currently supports SIMBA only, not {input_source!r}.'
        )

    selection = config['selection']
    selection_type = selection['type']
    if selection_type not in {'caesar_host_halo', 'caesar_galaxy'}:
        raise ValueError(
            f'Unknown SIMBA CAESAR selection type: {selection_type!r}'
        )
    gas_rows = simba.select_halo_gas_rows(
        resolve_path(selection['catalogue_filename']),
        int(selection['target_galaxy_idx']),
        find_host=(selection_type == 'caesar_host_halo'),
    )

    redshift = config.get('redshift')
    if redshift is None:
        redshift = simba.get_redshift(snapshot_filename)

    emissivity = config['emissivity']
    result = run_line_emission_pipeline(
        snapshot_filename,
        gas_rows,
        table_type=emissivity['type'],
        table_filename=resolve_path(emissivity['table_filename']),
        line_label=emissivity['line_label'],
        redshift=float(redshift),
        unit_factors=config['unit_factors'],
        center_kpc=config['center_kpc'],
        x_edges_kpc=config['x_edges_kpc'],
        y_edges_kpc=config['y_edges_kpc'],
        velocity_edges_kms=config['velocity_edges_kms'],
        los_axis=config.get('los_axis', 'z'),
        systemic_velocity_kms=config.get('systemic_velocity_kms', 0.0),
    )

    output_path = Path(resolve_path(config['output_filename']))
    if output_path.suffix != '.npz':
        output_path = Path(f'{output_path}.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    print(
        f"Saved {result['cube_erg_s'].shape} cube from "
        f"{result['selected_particle_count']} particles to {output_path}"
    )


if __name__ == '__main__':
    # Active path: CAESAR row indices; ParticleIDs are not used for selection.
    main_controller()
