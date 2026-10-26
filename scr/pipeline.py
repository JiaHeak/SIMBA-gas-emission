"""Main controller for the line-emission map pipeline."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    # Also allow: python Pipeline/scr/pipeline.py config.yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Pipeline.scr import simba
    from Pipeline.scr import input
    from Pipeline.scr.emissivity_tables import calculate_particle_line_luminosity
    from Pipeline.scr.ppv_cube import make_ppv_cube
    from Pipeline.scr.ppp_cube import make_ppp_cube
else:
    from . import simba
    from . import input
    from .emissivity_tables import calculate_particle_line_luminosity
    from .ppv_cube import make_ppv_cube
    from .ppp_cube import make_ppp_cube


def run_line_emission_pipeline(
    snapshot_filename, gas_rows, *, table_type, table_filename, line_label,
    redshift, unit_factors, center_kpc, x_edges_kpc, y_edges_kpc,
    velocity_edges_kms=None, los_axis='z', systemic_velocity_kms=0.0,
    box_size_kpc=None, cube_type='ppv', z_edges_kpc=None, grid_filename=None,
) -> dict:
    """Calculate particle emission once, then build the selected PPV or PPP cube.

    PPV is the default and requires velocity_edges_kms. PPP requires
    z_edges_kpc; it bins all selected velocities into a spatial volume.
    Centering and periodic offsets are shared by both output types.

    Returns: result (dict): luminosity cube, bin edges, and run metadata.
    """
    if cube_type not in ('ppv', 'ppp'):
        raise ValueError('cube_type must be ppv or ppp.')
    center = np.asarray(center_kpc, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError('center_kpc must contain three finite coordinates.')
    if box_size_kpc is not None:
        box_size_kpc = float(box_size_kpc)
        if not np.isfinite(box_size_kpc) or box_size_kpc <= 0:
            raise ValueError('box_size_kpc must be positive and finite.')

    bin_edges = []
    for name, values in (
        ('x_edges_kpc', x_edges_kpc),
        ('y_edges_kpc', y_edges_kpc),
        ('velocity_edges_kms', velocity_edges_kms) if cube_type == 'ppv'
        else ('z_edges_kpc', z_edges_kpc),
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
    x_edges_kpc, y_edges_kpc, third_edges = bin_edges

    redshift = float(redshift)
    if not np.isfinite(redshift):
        raise ValueError('redshift must be finite.')

    gas_fields = simba.load_physical_gas_fields(
        snapshot_filename, gas_rows, unit_factors
    )
    for axis_index, axis in enumerate(('x', 'y', 'z')):
        gas_fields[f'{axis}_kpc'] -= center[axis_index]
        if box_size_kpc is not None:
            gas_fields[f'{axis}_kpc'] = (
                gas_fields[f'{axis}_kpc'] + box_size_kpc / 2
            ) % box_size_kpc - box_size_kpc / 2

    table_options = {} if grid_filename is None else {'grid_filename': grid_filename}
    emission = calculate_particle_line_luminosity(
        gas_fields,
        table_type=table_type,
        table_filename=table_filename,
        redshift=redshift,
        line_label=line_label,
        **table_options,
    )
    if cube_type == 'ppv':
        result = make_ppv_cube(
            gas_fields, emission['line_luminosity_erg_s'],
            x_edges_kpc, y_edges_kpc, third_edges,
            los_axis=los_axis, systemic_velocity_kms=systemic_velocity_kms,
        )
        result['systemic_velocity_kms'] = float(systemic_velocity_kms)
        result['cube_axis_order'] = 'velocity,y,x'
    else:
        result = make_ppp_cube(
            gas_fields, emission['line_luminosity_erg_s'],
            x_edges_kpc, y_edges_kpc, third_edges,
        )
    result.update({
        'input_source': 'simba',
        'cube_type': cube_type,
        'table_type': table_type,
        'line_label': line_label,
        'redshift': redshift,
        'selected_particle_count': len(gas_rows),
        'center_kpc': center,
    })
    if box_size_kpc is not None:
        result['box_size_kpc'] = box_size_kpc
    return result


def main_controller():
    """Run the active row-based pipeline from a YAML configuration.

    Returns: none (NoneType): saves the luminosity cube to an NPZ file.
    """
    parser = argparse.ArgumentParser(description='Run the line-emission cube pipeline from a YAML config.')
    parser.add_argument('config', help='Path to the pipeline YAML configuration')
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = input.load_config(config_path)
    cube_type = config.get('cube_type', 'ppv')
    if cube_type not in ('ppv', 'ppp'):
        raise ValueError('cube_type must be ppv or ppp.')

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
    if 'type' in selection or not {'group_type', 'target_index'} <= selection.keys():
        raise ValueError('selection must use group_type (halo or galaxy) and target_index.')
    group_type = selection['group_type']
    target_index = selection['target_index']
    if group_type not in {'halo', 'galaxy'}:
        raise ValueError('selection.group_type must be halo or galaxy.')
    if not isinstance(target_index, int) or isinstance(target_index, bool) or target_index < 0:
        raise ValueError('selection.target_index must be a non-negative integer.')
    catalogue_filename = resolve_path(selection['catalogue_filename'])
    reference_frame = None
    catalogue_redshift = simba.get_catalogue_redshift(catalogue_filename)
    snapshot_redshift = simba.get_redshift(snapshot_filename)
    if not np.isclose(catalogue_redshift, snapshot_redshift, rtol=0, atol=1e-6):
        raise ValueError('Catalogue and snapshot have different redshifts.')
    gas_rows = simba.select_gas_rows(catalogue_filename, target_index, group_type=group_type)
    center = config.get('center_kpc', 'auto')
    systemic = config.get('systemic_velocity_kms', 0.0)
    auto_center = center is None or (isinstance(center, str) and center == 'auto')
    if auto_center or (cube_type == 'ppv' and (systemic is None or systemic == 'auto')):
        reference_frame = simba.get_reference_frame(
            catalogue_filename, target_index, group_type=group_type,
        )

    redshift = config.get('redshift')
    if redshift is None:
        redshift = snapshot_redshift

    emissivity = config['emissivity']
    cube_parameters = input.get_cube_parameters(config, reference_frame)
    box_size_kpc = None
    if config.get('periodic', False):
        box_size_kpc = input.get_physical_box_size(snapshot_filename, config['unit_factors']['position_to_physical_kpc'])

    table_options = {}
    if 'grid_filename' in emissivity:
        table_options['grid_filename'] = resolve_path(emissivity['grid_filename'])

    result = run_line_emission_pipeline(
        snapshot_filename,
        gas_rows,
        table_type=emissivity['type'],
        table_filename=resolve_path(emissivity['table_filename']),
        line_label=emissivity['line_label'],
        redshift=float(redshift),
        unit_factors=config['unit_factors'],
        box_size_kpc=box_size_kpc,
        **cube_parameters,
        **table_options)

    result.update({
        'config_filename': str(config_path),
        'snapshot_filename': snapshot_filename,
        'catalogue_filename': catalogue_filename,
        'group_type': group_type,
        'target_index': target_index,
    })

    if 'grid_filename' in table_options:
        result['grid_filename'] = table_options['grid_filename']

    output_filename = config.get('output_filename', 'auto')
    if output_filename is None or output_filename == 'auto':
        line_name = re.sub(r'[^A-Za-z0-9]+', '_', emissivity['line_label']).strip('_')
        prefix = 'line_cube' if cube_type == 'ppv' else 'line_cube_ppp'
        output_filename = f'{prefix}_{group_type}_{target_index}_{line_name}.npz'
    output_path = Path(resolve_path(output_filename))
    if output_path.suffix != '.npz':
        output_path = Path(f'{output_path}.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    print(
        f"Saved {cube_type.upper()} {result['cube_erg_s'].shape} cube from "
        f"{result['selected_particle_count']} particles to {output_path}"
    )


if __name__ == '__main__':
    # Catalogue membership supplies snapshot row indices.
    main_controller()
