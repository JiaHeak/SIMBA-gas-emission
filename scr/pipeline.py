"""Produce a reusable particle emission catalogue from one selected halo."""
import argparse
import sys
from pathlib import Path
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scr import simba, input
    from scr.emissivity_tables import load_emissivity_table, iter_particle_line_luminosities
    from scr.particle_output import save_particle_catalogue
else:
    from . import simba, input
    from .emissivity_tables import load_emissivity_table, iter_particle_line_luminosities
    from .particle_output import save_particle_catalogue


def prepare_run(parameters):
    """Resolve units, table and a default reference without choosing render bins.

    Returns: run (dict): shared physical units, table, redshift and reference frame.
    """
    source = parameters['source']
    snapshot = source['snapshot_filename']
    catalogue = source['catalogue_filename']
    model = parameters['emissivity']
    table = load_emissivity_table(model['type'], model['table_filename'], model['line_labels'])
    redshift = simba.get_redshift(snapshot)
    if not np.isclose(redshift, simba.get_catalogue_redshift(catalogue), rtol=0, atol=1e-6):
        raise ValueError('Snapshot and catalogue redshifts differ.')
    units = simba.compute_unit_factors(snapshot, parameters['unit_factors'])
    parameters['physical_unit_factors'] = units
    frame = dict(parameters['reference_frame'])
    halo_frame = simba.get_reference_frame(catalogue, **parameters['selection'])
    frame['halo_center_kpc'] = halo_frame['center_kpc']
    frame['halo_bulk_velocity_kms'] = halo_frame['bulk_velocity_kms']
    for name in ('center_kpc', 'bulk_velocity_kms'):
        if isinstance(frame[name], str) and frame[name] == 'auto':
            frame[name] = halo_frame[name]
    frame['box_size_kpc'] = input.get_physical_box_size(snapshot, units['position_to_physical_kpc'])
    run_redshift = redshift if source['redshift'] == 'auto' else source['redshift']
    for fixed in table['data'].get('radiation_redshifts', []):
        if not np.isclose(run_redshift, fixed, rtol=0, atol=1e-6):
            raise ValueError('Run and fixed Cloudy radiation redshifts differ.')
    return dict(unit_factors=units, frame=frame, table=table, redshift=run_redshift)


def main(config_filename=None):
    """Select gas, calculate emission once and save particles for later rendering.

    Returns: output_path (pathlib.Path): complete particle emission catalogue.
    """
    if config_filename is None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('config', help='YAML configuration filename')
        config_filename = parser.parse_args().config
    parameters = input.read_parameters(config_filename)
    run = prepare_run(parameters)
    source = parameters['source']
    halo_rows = simba.select_gas_rows(source['catalogue_filename'], **parameters['selection'])
    rows = simba.select_cgm_gas_rows(source['snapshot_filename'], source['catalogue_filename'],
                                   halo_rows, **parameters['selection'], unit_factors=run['unit_factors'])
    gas = simba.load_catalogue_gas_fields(source['snapshot_filename'], rows, run['unit_factors'])
    emissions = iter_particle_line_luminosities(gas, emissivity_table=run['table'], redshift=run['redshift'])
    return save_particle_catalogue(gas, emissions, parameters, run)


if __name__ == '__main__':
    main()
