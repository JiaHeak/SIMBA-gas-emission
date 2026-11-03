# Here we define those functions used to read inputs

from pathlib import Path

import h5py
import numpy as np
import yaml


def load_config(config_filename) -> dict:
    """Read a YAML settings file without constructing Python objects.

    Returns: config (dict): settings keyed by their YAML parameter names.
    """
    with open(config_filename, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping of parameter names to values.')
    return config


def read_parameters(config_filename) -> dict:
    """Read grouped YAML parameters, apply defaults, and resolve file paths.

    No simulation data are read here. unit_factors contains global constants;
    simba.compute_unit_factors() resolves the snapshot-dependent factors later.
    Catalogue-derived values remain 'auto' until run preparation. Paths use the YAML folder,
    not the shell's working directory. Unknown keys raise rather than disappear.

    Returns: parameters (dict): source, selection, unit_factors, emissivity, cube,
        output, and the absolute config_filename used for provenance.
    """
    config_path = Path(config_filename).expanduser().resolve()
    config = load_config(config_path)

    def section(name, required, defaults=None):
        """Check a section and fill its optional values.

        Returns: values (dict): copied settings with explicit defaults.
        """
        values = config.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f'{name} must be a YAML mapping.')
        defaults = {} if defaults is None else defaults
        missing = set(required) - values.keys()
        unknown = values.keys() - (set(required) | defaults.keys())
        if missing or unknown:
            raise ValueError(f'{name}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}.')
        return {**defaults, **values}

    expected = {'source', 'selection', 'unit_factors', 'emissivity', 'cube', 'output'}
    if config.keys() - expected:
        raise ValueError('Use grouped YAML sections: source, selection, unit_factors, '
                         'emissivity, cube, output. Unknown top-level keys: '
                         f'{sorted(config.keys() - expected)}')
    source = section('source', {'snapshot_filename', 'catalogue_filename'},
                     {'type': 'simba', 'redshift': 'auto'})
    selection = section('selection', {'group_type', 'target_index'})
    units = section('unit_factors', {
        'solar_mass_g', 'kpc_to_cm', 'specific_energy_to_erg_g', 'z_sun_mass_fraction',
    }, {'gamma': 5.0 / 3.0})
    emissivity = section('emissivity', {'type', 'table_filename', 'line_labels'})
    cube = section('cube', set(), {
        'type': 'ppv', 'center_kpc': 'auto', 'periodic': False,
        'map_half_width_kpc': 100, 'spatial_bins': 2,
        'velocity_half_width_kms': 300, 'velocity_bins': 2,
        'los_axis': 'z', 'systemic_velocity_kms': 0.0,
        'x_edges_kpc': None, 'y_edges_kpc': None, 'z_edges_kpc': None,
        'velocity_edges_kms': None,
    })
    output = section('output', set(), {'filename': 'auto'})

    def number(value, name, positive=False):
        """Reject nonnumeric or nonfinite physical parameters.

        Returns: value (float): validated numerical parameter.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a number, not {value!r}.')
        if not np.isfinite(value) or (positive and value <= 0):
            raise ValueError(f'{name} must be finite' + (' and positive.' if positive else '.'))
        return float(value)

    if source['type'] != 'simba':
        raise ValueError('source.type currently supports simba only.')
    if source['redshift'] != 'auto':
        source['redshift'] = number(source['redshift'], 'source.redshift')
    if selection['group_type'] != 'halo':
        raise ValueError('selection.group_type must be halo: current CGM selection is halo-only.')
    index = selection['target_index']
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError('selection.target_index must be a non-negative integer.')
    for name, value in units.items():
        units[name] = number(value, f'unit_factors.{name}', positive=True)
    if units['gamma'] <= 1:
        raise ValueError('unit_factors.gamma must be greater than one.')
    if units['z_sun_mass_fraction'] >= 1:
        raise ValueError('unit_factors.z_sun_mass_fraction must be below one.')
    if emissivity['type'] not in ('ps20', 'cloudy_hdf5'):
        raise ValueError('emissivity.type must be ps20 or cloudy_hdf5 (HDF5 tables only).')
    labels = emissivity['line_labels']
    if (not isinstance(labels, list) or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)):
        raise ValueError('emissivity.line_labels must be a nonempty list of exact table identifiers.')
    if len(set(labels)) != len(labels):
        raise ValueError('emissivity.line_labels contains repeated labels.')
    if cube['type'] not in ('ppv', 'ppp'):
        raise ValueError('cube.type must be ppv or ppp.')
    if not isinstance(cube['periodic'], bool):
        raise ValueError('cube.periodic must be a YAML boolean (true or false).')
    center = cube['center_kpc']
    if not (isinstance(center, str) and center == 'auto'):
        if not isinstance(center, list) or len(center) != 3:
            raise ValueError('cube.center_kpc must be auto or a list of three finite coordinates.')
        cube['center_kpc'] = [number(value, 'cube.center_kpc') for value in center]
    systemic = cube['systemic_velocity_kms']
    if cube['type'] == 'ppv' and systemic != 'auto':
        cube['systemic_velocity_kms'] = number(systemic, 'cube.systemic_velocity_kms')
    # Keep the existing cube-builder keyword; YAML uses the shorter cube.type.
    cube['cube_type'] = cube.pop('type')

    def resolve_path(value, name):
        """Resolve an explicitly supplied input or output filename.

        Returns: filename (str): absolute path; existence is checked by readers.
        """
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty file path.')
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        return str(path.resolve())

    for key in ('snapshot_filename', 'catalogue_filename'):
        source[key] = resolve_path(source[key], f'source.{key}')
    emissivity['table_filename'] = resolve_path(emissivity['table_filename'], 'emissivity.table_filename')
    if output['filename'] == 'auto':
        output['filename'] = f"line_cubes_{cube['cube_type']}_halo_{index}.hdf5"
    output['filename'] = resolve_path(output['filename'], 'output.filename')
    suffix = Path(output['filename']).suffix.lower()
    if not suffix:
        output['filename'] += '.hdf5'
    elif suffix not in ('.hdf5', '.h5'):
        raise ValueError('output.filename must end in .hdf5 or .h5; NPZ cube output is no longer supported.')
    return {
        'source': source, 'selection': selection, 'unit_factors': units,
        'emissivity': emissivity, 'cube': cube, 'output': output,
        'config_filename': str(config_path),
    }


def get_physical_box_size(snapshot_filename: str, position_to_physical_kpc: float) -> float:
    """Convert SIMBA Header/BoxSize with the same factor as gas coordinates.

    Returns: box_size_kpc (float): positive physical box side length in kpc.
    """
    with h5py.File(snapshot_filename, 'r') as snapshot:
        box_size = float(snapshot['Header'].attrs['BoxSize']) * position_to_physical_kpc
    if not np.isfinite(box_size) or box_size <= 0:
        raise ValueError('Physical box size must be positive and finite.')
    return box_size


def get_cube_parameters(config: dict, reference_frame=None) -> dict:
    """Resolve the center and the edges needed by the chosen cube builder.

    Explicit edges override generated uniform bins. PPV uses two image axes
    and velocity; PPP uses physical x, y, z offsets and no velocity selection.

    Returns: parameters (dict): cube type, physical center, and builder settings.
    """
    cube_type = config.get('cube_type', 'ppv')
    if cube_type not in ('ppv', 'ppp'):
        raise ValueError('cube_type must be ppv or ppp.')
    center = config.get('center_kpc', 'auto')
    if center is None or (isinstance(center, str) and center == 'auto'):
        if reference_frame is None:
            raise ValueError('Automatic center requires a catalogue reference frame.')
        center = reference_frame['center_kpc']

    def get_edges(name, width_key, count_key, default_width):
        """Read explicit edges or generate uniform bins.

        Returns: edges (numpy.ndarray): finite, strictly increasing bin edges.
        """
        if config.get(name) is not None:
            edges = np.asarray(config[name], dtype=np.float64)
        else:
            width = float(config.get(width_key, default_width))
            count = config.get(count_key, 2)
            if not np.isfinite(width) or width <= 0:
                raise ValueError(f'{width_key} must be positive and finite.')
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(f'{count_key} must be a positive integer.')
            edges = np.linspace(-width, width, count + 1)
        if (edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges))
                or not np.all(np.diff(edges) > 0)):
            raise ValueError(f'{name} must be finite, 1D, and strictly increasing.')
        return edges

    parameters = {
        'cube_type': cube_type,
        'center_kpc': center,
        'x_edges_kpc': get_edges('x_edges_kpc', 'map_half_width_kpc', 'spatial_bins', 100),
        'y_edges_kpc': get_edges('y_edges_kpc', 'map_half_width_kpc', 'spatial_bins', 100),
    }
    if cube_type == 'ppp':
        parameters['z_edges_kpc'] = get_edges(
            'z_edges_kpc', 'map_half_width_kpc', 'spatial_bins', 100,
        )
    else:
        los_axis = config.get('los_axis', 'z')
        if los_axis not in ('x', 'y', 'z'):
            raise ValueError('los_axis must be x, y, or z.')
        systemic = config.get('systemic_velocity_kms', 0.0)
        if systemic is None or systemic == 'auto':
            if reference_frame is None:
                raise ValueError('Automatic systemic velocity requires a catalogue reference frame.')
            systemic = reference_frame['bulk_velocity_kms'][('x', 'y', 'z').index(los_axis)]
        parameters.update({
            'los_axis': los_axis,
            'systemic_velocity_kms': float(systemic),
            'velocity_edges_kms': get_edges(
                'velocity_edges_kms', 'velocity_half_width_kms', 'velocity_bins', 300,
            ),
        })
    return parameters


