# Here we define those functions used to read inputs

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


