"""Report whether selected gas falls inside a sample cube's configured bins."""

import argparse
from pathlib import Path

import numpy as np

from Pipeline.scr import simba
from Pipeline.scr import input as pipeline_input
from Pipeline.scr.pipeline import prepare_run


def report_cube_coverage(config_filename: str) -> None:
    """Print CGM ranges/counts without calculating emission or writing cubes.

    Shared run preparation validates the requested table/line metadata first.

    Returns: none (NoneType): prints a compact diagnostic report.
    """
    config = pipeline_input.read_parameters(config_filename)
    if config['cube']['cube_type'] != 'ppv':
        raise ValueError(
            'This coverage diagnostic supports PPV only. '
            'PPP output includes included/excluded particle and luminosity totals.'
        )

    run = prepare_run(config)
    snapshot_filename = config['source']['snapshot_filename']
    gas_rows = simba.select_gas_rows(**run['halo_selection'])
    if gas_rows.size == 0:
        raise ValueError('Cannot report coordinate ranges for an empty gas selection.')
    gas_rows = simba.select_cgm_gas_rows(gas_rows=gas_rows, **run['cgm_selection'])
    if gas_rows.size == 0:
        raise ValueError('Cannot report coordinate ranges for an empty CGM selection.')
    parameters = run['calculation']
    gas = simba.load_physical_gas_fields(
        snapshot_filename, gas_rows, parameters['unit_factors']
    )
    center = np.asarray(parameters['center_kpc'], dtype=np.float64)
    axis = parameters['los_axis']
    image_keys = {
        'x': ('y_kpc', 'z_kpc', 'v_x_kms'),
        'y': ('x_kpc', 'z_kpc', 'v_y_kms'),
        'z': ('x_kpc', 'y_kpc', 'v_z_kms'),
    }
    image_x_key, image_y_key, velocity_key = image_keys[axis]
    coordinate_index = {'x_kpc': 0, 'y_kpc': 1, 'z_kpc': 2}
    image_x = gas[image_x_key] - center[coordinate_index[image_x_key]]
    image_y = gas[image_y_key] - center[coordinate_index[image_y_key]]
    if parameters['box_size_kpc'] is not None:
        box_size = parameters['box_size_kpc']
        image_x = (image_x + box_size / 2) % box_size - box_size / 2
        image_y = (image_y + box_size / 2) % box_size - box_size / 2
    velocity = gas[velocity_key] - parameters['systemic_velocity_kms']
    axes = (
        ('image_x_kpc', image_x, np.asarray(parameters['x_edges_kpc'])),
        ('image_y_kpc', image_y, np.asarray(parameters['y_edges_kpc'])),
        ('line_of_sight_velocity_kms', velocity, np.asarray(parameters['velocity_edges_kms'])),
    )
    print('selected_rows:', len(gas_rows))
    print('zero_metallicity_rows:', int(np.count_nonzero(gas['Z_Zsun'] == 0)))
    for name, values, edges in axes:
        print(f'{name}_range: [{values.min()}, {values.max()}]')
        print(f'{name}_configured_edges: [{edges[0]}, {edges[-1]}]')
        print(f'{name}_within_edges: {int(np.count_nonzero((values >= edges[0]) & (values <= edges[-1])))}')
    counts, _ = np.histogramdd(
        np.column_stack((image_x, image_y, velocity)),
        bins=(parameters['x_edges_kpc'], parameters['y_edges_kpc'], parameters['velocity_edges_kms']),
    )
    print('rows_inside_all_cube_bins:', int(counts.sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config_filename', type=Path)
    args = parser.parse_args()
    report_cube_coverage(str(args.config_filename))
