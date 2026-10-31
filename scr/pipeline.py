"""Main controller for the line-emission map pipeline."""

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    # Also allow: python Pipeline/scr/pipeline.py config.yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Pipeline.scr import simba
    from Pipeline.scr import input
    from Pipeline.scr.emissivity_tables import load_emissivity_table, iter_particle_line_luminosities
    from Pipeline.scr.output import save_result
    from Pipeline.scr.ppv_cube import make_ppv_cube
    from Pipeline.scr.ppp_cube import make_ppp_cube
else:
    from . import simba
    from . import input
    from .emissivity_tables import load_emissivity_table, iter_particle_line_luminosities
    from .output import save_result
    from .ppv_cube import make_ppv_cube
    from .ppp_cube import make_ppp_cube


def run_line_emission_pipeline(
    snapshot_filename, gas_rows, *, emissivity_table,
    redshift, unit_factors, center_kpc, x_edges_kpc, y_edges_kpc,
    velocity_edges_kms=None, los_axis='z', systemic_velocity_kms=0.0,
    box_size_kpc=None, cube_type='ppv', z_edges_kpc=None,
):
    """Read/center physical gas once, then yield a cube for every requested line.

    PPV is the default and requires velocity_edges_kms. PPP requires
    z_edges_kpc; it bins all selected velocities into a spatial volume.
    Centering and periodic offsets are shared by both output types.

    Returns: results (collections.abc.Iterator): ordered cube dictionaries with
        shared geometry and per-line luminosity, coverage, and clipping metadata.
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

    for emission in iter_particle_line_luminosities(
        gas_fields, emissivity_table=emissivity_table, redshift=redshift,
    ):
        if cube_type == 'ppv':
            result = make_ppv_cube(
                gas_fields, emission['line_luminosity_erg_s'],
                x_edges_kpc, y_edges_kpc, third_edges,
                los_axis=los_axis, systemic_velocity_kms=systemic_velocity_kms,
            )
            result['systemic_velocity_kms'] = float(systemic_velocity_kms)
        else:
            result = make_ppp_cube(
                gas_fields, emission['line_luminosity_erg_s'],
                x_edges_kpc, y_edges_kpc, third_edges,
            )
        result.update({
            'input_source': 'simba',
            'cube_type': cube_type,
            'table_type': emissivity_table['table_type'],
            'table_filename': emissivity_table['table_filename'],
            'table_axes': emissivity_table['data']['axes'],
            'line_label': emission['line_label'],
            'redshift': redshift,
            'selected_particle_count': len(gas_rows),
            'center_kpc': center,
        })
        for name in ('abundance_model', 'clipped_particle_count'):
            if name in emission:
                result[name] = emission[name]
        if box_size_kpc is not None:
            result['box_size_kpc'] = box_size_kpc
        yield result


def prepare_run(parameters: dict) -> dict:
    """Resolve simulation-derived settings without loading particle fields.

    Returns: run (dict): arguments for halo selection, CGM selection, and
        the existing line-emission calculation.
    """
    source = parameters['source']
    selection = parameters['selection']
    units = parameters['unit_factors']
    cube = parameters['cube']
    emissivity = parameters['emissivity']
    emissivity_table = load_emissivity_table(
        emissivity['type'], emissivity['table_filename'], emissivity['line_labels'],
    )
    snapshot_filename = source['snapshot_filename']
    catalogue_filename = source['catalogue_filename']

    snapshot_redshift = simba.get_redshift(snapshot_filename)
    catalogue_redshift = simba.get_catalogue_redshift(catalogue_filename)
    if not np.isclose(catalogue_redshift, snapshot_redshift, rtol=0, atol=1e-6):
        raise ValueError('Catalogue and snapshot have different redshifts.')

    reference_frame = None
    center = cube['center_kpc']
    auto_center = center is None or (isinstance(center, str) and center == 'auto')
    auto_velocity = cube['systemic_velocity_kms'] in (None, 'auto')
    if auto_center or (cube['cube_type'] == 'ppv' and auto_velocity):
        reference_frame = simba.get_reference_frame(catalogue_filename, **selection)
    cube_parameters = input.get_cube_parameters(cube, reference_frame)
    box_size_kpc = None
    if cube['periodic']:
        box_size_kpc = input.get_physical_box_size(
            snapshot_filename, units['position_to_physical_kpc'],
        )
    redshift = snapshot_redshift if source['redshift'] == 'auto' else source['redshift']
    calculation = {
        'snapshot_filename': snapshot_filename, 'unit_factors': units,
        'redshift': redshift, 'box_size_kpc': box_size_kpc,
        'emissivity_table': emissivity_table,
        **cube_parameters,
    }
    return {
        # **selection adds group_type and target_index to the argument dictionaries.
        'halo_selection': {'catalogue_filename': catalogue_filename, **selection},
        'cgm_selection': {
            'snapshot_filename': snapshot_filename,
            'catalogue_filename': catalogue_filename, 'unit_factors': units,
            **selection,
        },
        'calculation': calculation,
    }


def parse_command_line() -> str:
    """Read the single YAML filename supplied at the command line.

    Returns: config_filename (str): user-supplied YAML path.
    """
    parser = argparse.ArgumentParser(description='Run the line-emission cube pipeline from a YAML config.')
    parser.add_argument('config', help='Path to the pipeline YAML configuration')
    return parser.parse_args().config


def main(config_filename=None):
    """Run the workflow using grouped, validated YAML parameters.

    Returns: none (NoneType): saves all requested line cubes in one HDF5 file.
    """
    if config_filename is None:
        config_filename = parse_command_line()
    parameters = input.read_parameters(config_filename)
    run = prepare_run(parameters)
    halo_rows = simba.select_gas_rows(**run['halo_selection'])
    cgm_rows = simba.select_cgm_gas_rows(gas_rows=halo_rows, **run['cgm_selection'])
    results = run_line_emission_pipeline(gas_rows=cgm_rows, **run['calculation'])
    save_result(results, parameters)


if __name__ == '__main__':
    main()
