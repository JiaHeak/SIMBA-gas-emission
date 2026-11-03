"""Atomic, chunked HDF5 storage for reusable particle emission results."""
import json
import os
from pathlib import Path
import tempfile
import h5py
import numpy as np
import yaml

SCHEMA = 'codex_simba_particle_emission_v1'


def _json(value):
    """Serialize small numerical metadata without losing its named structure.

    Returns: text (str): JSON with NumPy scalars/arrays represented as numbers/lists.
    """
    return json.dumps(value, default=lambda x: x.tolist(), allow_nan=False)


def save_particle_catalogue(gas, emissions, parameters, run):
    """Store shared particles once and stream each requested line to disk.

    Positions are physical box coordinates; velocities are physical peculiar
    velocities. Reference-frame subtraction is deferred to the Reader.
    Returns: output_path (pathlib.Path): newly published complete HDF5 catalogue.
    """
    path = Path(parameters['output']['filename'])
    if path.exists():
        raise FileExistsError(path)
    arrays = {
        'snapshot_row': np.asarray(gas['snapshot_row']),
        'particle_id': np.asarray(gas['particle_id']),
        'position_kpc': np.column_stack([gas[f'{a}_kpc'] for a in 'xyz']),
        'velocity_kms': np.column_stack([gas[f'v_{a}_kms'] for a in 'xyz']),
        'volume_cm3': np.asarray(gas['volume_cm3']),
        'smoothing_length_kpc': np.asarray(gas['smoothing_length_kpc']),
    }
    units = dict(snapshot_row='dimensionless', particle_id='dimensionless', position_kpc='kpc',
                 velocity_kms='km s^-1', volume_cm3='cm^3', smoothing_length_kpc='kpc')
    n = len(arrays['snapshot_row'])
    for name, values in arrays.items():
        shape = (n, 3) if name in ('position_kpc', 'velocity_kms') else (n,)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f'Invalid or misaligned particle field: {name}')
    for name in ('snapshot_row', 'particle_id'):
        if not np.issubdtype(arrays[name].dtype, np.integer) or np.any(arrays[name] < 0):
            raise ValueError(f'{name} must contain nonnegative integers.')
    if len(np.unique(arrays['snapshot_row'])) != n:
        raise ValueError('Repeated snapshot rows; ParticleIDs themselves may repeat.')
    for name in ('volume_cm3', 'smoothing_length_kpc'):
        if np.any(arrays[name] <= 0):
            raise ValueError(f'{name} must be positive.')
    frame = run['frame']
    for name in ('center_kpc', 'bulk_velocity_kms', 'halo_center_kpc', 'halo_bulk_velocity_kms'):
        if np.shape(frame[name]) != (3,) or not np.all(np.isfinite(frame[name])):
            raise ValueError(f'Invalid reference {name}.')
    if not np.isfinite(frame['box_size_kpc']) or frame['box_size_kpc'] <= 0:
        raise ValueError('Box size must be positive and finite.')
    labels = parameters['emissivity']['line_labels']
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Request nonempty, unique line labels.')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.particle_emission_', suffix='.hdf5')
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as out:
            out.attrs.update(schema=SCHEMA, complete=False, particle_count=n,
                             line_count=len(labels), redshift=run['redshift'])
            particles = out.create_group('particles')
            for name, values in arrays.items():
                dataset = particles.create_dataset(name, data=values, chunks=True,
                                                    compression='gzip', shuffle=True)
                dataset.attrs['unit'] = units[name]
            particles['position_kpc'].attrs.update(frame='physical simulation-box coordinates', component_order='x,y,z')
            particles['velocity_kms'].attrs.update(frame='physical peculiar velocity; simulation frame', component_order='x,y,z')
            particles['smoothing_length_kpc'].attrs['definition'] = 'GIZMO kernel support radius; no rendering kernel applied'
            reference = out.create_group('reference_frame')
            for name, unit in [('center_kpc', 'kpc'), ('bulk_velocity_kms', 'km s^-1'),
                               ('halo_center_kpc', 'kpc'), ('halo_bulk_velocity_kms', 'km s^-1'),
                               ('box_size_kpc', 'kpc')]:
                reference.create_dataset(name, data=frame[name]).attrs['unit'] = unit
            reference.attrs['periodic'] = frame['periodic']
            provenance = out.create_group('provenance')
            provenance.create_dataset('parameters_yaml', data=yaml.safe_dump(parameters, sort_keys=False))
            provenance.create_dataset('unit_factors_json', data=_json(run['unit_factors']))
            provenance.attrs['selection'] = 'Octavius halo gas; radius <= r200m and nH <= 0.13 cm^-3; periodic radial distance'
            model = run['table']
            metadata = {key: value for key, value in model['data'].items()
                        if key != 'volume_emissivity_erg_s_cm3'}
            provenance.create_dataset('table_metadata_json', data=_json(metadata))
            provenance.attrs.update(table_type=model['table_type'], table_filename=model['table_filename'],
                                    snapshot_filename=parameters['source']['snapshot_filename'],
                                    catalogue_filename=parameters['source']['catalogue_filename'],
                                    target_index=parameters['selection']['target_index'])
            lines = out.create_group('lines', track_order=True)
            lines.create_dataset('labels', data=labels, dtype=h5py.string_dtype())
            count = 0
            for index, emission in enumerate(emissions):
                if index >= len(labels) or emission['line_label'] != labels[index]:
                    raise ValueError('Emission line order differs from configuration.')
                epsilon = np.asarray(emission['volume_emissivity_erg_s_cm3'])
                luminosity = np.asarray(emission['line_luminosity_erg_s'])
                for values in (epsilon, luminosity):
                    if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                        raise ValueError('Emission must be finite, nonnegative and aligned by particle.')
                if not np.allclose(luminosity, epsilon * arrays['volume_cm3'], rtol=1e-12, atol=0):
                    raise ValueError('Particle luminosity does not equal emissivity times volume.')
                line = lines.create_group(f'line_{index:04d}')
                line.attrs['line_label'] = labels[index]
                for name, values, unit in [('volume_emissivity_erg_s_cm3', epsilon, 'erg s^-1 cm^-3'),
                                           ('luminosity_erg_s', luminosity, 'erg s^-1')]:
                    line.create_dataset(name, data=values, chunks=True, compression='gzip', shuffle=True).attrs['unit'] = unit
                diagnostic = {key: value for key, value in emission.items()
                              if key not in ('volume_emissivity_erg_s_cm3', 'line_luminosity_erg_s', 'log_volume_emissivity')}
                line.attrs['diagnostics_json'] = _json(diagnostic)
                count += 1
            if count != len(labels):
                raise ValueError('Missing requested emission lines.')
            out.attrs['complete'] = True
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f'Saved {n} particles and {count} lines to {path}')
    return path
