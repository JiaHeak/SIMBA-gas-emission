"""Write ordered line cubes to the project's versioned HDF5 output format.

Version 2 arrays use (x,y,z) for PPP and (image x,image y,velocity) for PPV.
Shared coordinates live in /axes and /geometry. /lines/labels preserves YAML
order; /lines/line_0000, line_0001, ... each hold one line's cubes/diagnostics.
Labels are attributes rather than HDF5 paths, so spaces or slashes are safe.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import yaml

SCHEMA = 'codex_simba_line_cubes_v2'


def _write_shared(destination, result: dict, parameters: dict) -> None:
    """Write the geometry and run provenance shared by all requested lines.

    Returns: none (NoneType): creates common HDF5 groups, datasets and attributes.
    """
    destination.attrs['schema'] = SCHEMA
    destination.attrs['created_utc'] = datetime.now(timezone.utc).isoformat()
    for name in ('cube_type', 'cube_axis_order', 'input_source', 'table_type',
                 'redshift', 'selected_particle_count', 'image_x_axis',
                 'image_y_axis', 'los_axis', 'systemic_velocity_kms'):
        if name in result:
            destination.attrs[name] = result[name]
    destination.attrs['deposition'] = result.get('deposition', 'point deposition; no smoothing or line broadening')
    for name in ('smoothing_enabled', 'smoothing_kernel_1d', 'smoothing_sigma_voxels',
                 'smoothing_boundary', 'smoothing_caveat'):
        if name in result:
            destination.attrs[name] = result[name]
    axes = destination.create_group('axes')
    for name, values in result.items():
        if name.endswith('_edges_kpc') or name == 'velocity_edges_kms':
            dataset = axes.create_dataset(name, data=values)
            dataset.attrs['unit'] = 'km s^-1' if name == 'velocity_edges_kms' else 'kpc'
            dataset.attrs['frame'] = 'relative to systemic velocity' if name == 'velocity_edges_kms' else 'physical offsets from center'
    geometry = destination.create_group('geometry')
    center = geometry.create_dataset('center_kpc', data=result['center_kpc'])
    center.attrs['unit'] = 'kpc'
    center.attrs['axis_order'] = 'x,y,z'
    if 'box_size_kpc' in result:
        geometry.attrs['box_size_kpc'] = result['box_size_kpc']
    if 'voxel_volume_cm3' in result:
        volume = geometry.create_dataset('voxel_volume_cm3', data=result['voxel_volume_cm3'],
                                         compression='gzip', shuffle=True)
        volume.attrs['unit'] = 'cm^3'
        volume.attrs['axis_order'] = result['cube_axis_order']

    provenance = destination.create_group('provenance')
    provenance.attrs['config_filename'] = parameters['config_filename']
    for name in ('snapshot_filename', 'catalogue_filename'):
        provenance.attrs[name] = parameters['source'][name]
    provenance.attrs['table_filename'] = parameters['emissivity']['table_filename']
    for name, value in parameters['selection'].items():
        provenance.attrs[name] = value
    # Capture the effective input dictionary, not a YAML file that might later change.
    provenance.create_dataset('parameters_yaml', data=yaml.safe_dump(parameters, sort_keys=False),
                              dtype=h5py.string_dtype('utf-8'))
    units = provenance.create_group('unit_factors')
    # Standalone writer calls may already supply final factors. YAML constants
    # must be resolved by prepare_run() before they can describe a physical run.
    resolved_units = parameters.get('physical_unit_factors', parameters['unit_factors'])
    if 'solar_mass_g' in resolved_units:
        raise ValueError('Resolve YAML unit constants with prepare_run() before saving a datacube.')
    for name, value in resolved_units.items():
        if name in ('cosmological_factors', 'unit_conversions'):
            stage = provenance.create_group(name)
            for quantity, factor in value.items():
                stage.attrs[quantity] = factor
            stage.attrs['description'] = (
                'Dimensionless corrections from a and h only.' if name == 'cosmological_factors'
                else 'Unit-scale conversions only; independent of a and h.')
        else:
            units.attrs[name] = value
    if 'physical_unit_factors' in parameters:
        inputs = provenance.create_group('unit_inputs')
        for name, value in parameters['unit_factors'].items():
            inputs.attrs[name] = value
    table_axes = provenance.create_group('table_axes')
    axis_units = {'log_temperature_K': 'log10(K)', 'log_nH_cm3': 'log10(cm^-3)',
                  'log_metallicity_Zsun': 'log10(Z/Zsun)', 'redshift': 'dimensionless',
                  'log_metal_scale': 'dex'}
    for name, values in result['table_axes'].items():
        dataset = table_axes.create_dataset(name, data=values)
        if 'table_axis_metadata' in result:
            for key, value in result['table_axis_metadata'][name].items():
                dataset.attrs[key] = value
        else:
            # PS20 and standalone fixtures retain their established definitions.
            dataset.attrs['unit'] = axis_units[name]
    if 'table_schema' in result:
        table = provenance.create_group('emissivity_table')
        table.attrs['schema'] = result['table_schema']
        table.attrs['axis_order'] = ', '.join(result['table_axes'])
        table.attrs['abundance_model'] = result['abundance_model']
        table.create_dataset('physics_commands_json',
                             data=json.dumps(result['table_physics'], indent=2),
                             dtype=h5py.string_dtype('utf-8'))
        if 'interpolation_method' in result:
            table.attrs['interpolation_method'] = result['interpolation_method']
            table.attrs['metallicity_clipping_caveat'] = result['metallicity_clipping_caveat']
            # CAVEAT: this records the enriched-boundary approximation for Z=0;
            # provenance must not imply that primordial spectra were computed.
            table.create_dataset('metallicity_mapping_json',
                                 data=json.dumps(result['metallicity_mapping'], indent=2),
                                 dtype=h5py.string_dtype('utf-8'))


def save_result(results, parameters: dict) -> Path:
    """Stream all line cubes into one HDF5 file, publishing only a complete run.

    Existing files are not overwritten. A failed run removes its temporary
    output and leaves no apparently complete final product. Only one line's
    cube needs to be in memory at a time; no NPZ cube output is supported.

    Returns: output_path (pathlib.Path): path of the complete structured HDF5 file.
    """
    output_path = Path(parameters['output']['filename'])
    if output_path.suffix.lower() not in ('.hdf5', '.h5'):
        raise ValueError('Datacube output must be .hdf5 or .h5.')
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = parameters['emissivity']['line_labels']
    descriptor, temporary_name = tempfile.mkstemp(prefix='.line_cubes_', suffix='.hdf5', dir=output_path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    count = 0
    try:
        with h5py.File(temporary_path, 'w') as destination:
            lines = destination.create_group('lines', track_order=True)
            lines.create_dataset('labels', data=labels, dtype=h5py.string_dtype('utf-8'))
            lines.attrs['group_template'] = 'line_{index:04d}; index follows labels/YAML order'
            for index, result in enumerate(results):
                if index >= len(labels) or result['line_label'] != labels[index]:
                    raise ValueError('Cube results do not match the requested line order.')
                cube = np.asarray(result['cube_erg_s'])
                if cube.ndim != 3 or not np.all(np.isfinite(cube)) or np.any(cube < 0):
                    raise ValueError('Each line cube must be finite, nonnegative, and three-dimensional.')
                cube_type = result['cube_type']
                expected_order = {'ppp': 'x,y,z', 'ppv': 'x,y,velocity'}
                if cube_type not in expected_order or result['cube_axis_order'] != expected_order[cube_type]:
                    raise ValueError('Version 2 requires x,y,z (PPP) or x,y,velocity (PPV) axis order.')
                edge_names = (('x_edges_kpc', 'y_edges_kpc', 'z_edges_kpc') if cube_type == 'ppp'
                              else ('image_x_edges_kpc', 'image_y_edges_kpc', 'velocity_edges_kms'))
                expected_shape = tuple(len(result[name]) - 1 for name in edge_names)
                if cube.shape != expected_shape:
                    raise ValueError('Cube shape does not match its v2 coordinate edges.')
                if cube_type == 'ppp' and np.shape(result['voxel_volume_cm3']) != cube.shape:
                    raise ValueError('PPP voxel volumes must match the x,y,z cube shape.')
                if 'volume_emissivity_erg_s_cm3' in result and np.shape(result['volume_emissivity_erg_s_cm3']) != cube.shape:
                    raise ValueError('Emissivity shape must match the luminosity cube.')
                if index == 0:
                    _write_shared(destination, result, parameters)
                    cube_shape = cube.shape
                    saved_cube_type = cube_type
                elif cube.shape != cube_shape or cube_type != saved_cube_type:
                    raise ValueError('All line cubes must use the same geometry and cube type.')
                group = lines.create_group(f'line_{index:04d}')
                group.attrs['line_label'] = result['line_label']
                group.attrs['line_index'] = index
                for name, unit in (('cube_erg_s', 'erg s^-1'),
                                   ('volume_emissivity_erg_s_cm3', 'erg s^-1 cm^-3')):
                    if name in result:
                        dataset = group.create_dataset(name, data=result[name], compression='gzip', shuffle=True)
                        dataset.attrs['unit'] = unit
                        dataset.attrs['axis_order'] = result['cube_axis_order']
                        dataset.attrs['quantity'] = ('luminosity per voxel' if name == 'cube_erg_s'
                                                     else 'luminosity divided by full voxel volume')
                for name in ('input_particle_count', 'included_particle_count', 'excluded_particle_count',
                             'input_luminosity_erg_s', 'included_luminosity_erg_s', 'excluded_luminosity_erg_s',
                             'clipped_particle_count', 'abundance_model',
                             'zero_metallicity_particle_count', 'metallicity_clipped_particle_count',
                             'zero_emissivity_particle_count', 'interpolation_method',
                             'metallicity_clipping_caveat'):
                    if name in result:
                        group.attrs[name] = result[name]
                if 'clipping_counts' in result:
                    group.attrs['clipping_counts_json'] = json.dumps(result['clipping_counts'])
                count += 1
            if count != len(labels) or count == 0:
                raise ValueError('The run did not produce all requested line cubes.')
            destination.attrs['line_count'] = count
            destination.attrs['complete'] = True
        # Same-folder hard link is atomic and refuses a concurrent overwrite.
        os.link(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f'Saved {count} line cubes to {output_path}')
    return output_path
