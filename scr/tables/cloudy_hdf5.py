"""Interpolate fixed-abundance, one-zone Cloudy HDF5 tables with one or more lines."""

import warnings

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Historical format identifier, shared by the original MVP and multi-line tables.
# It defines the T/nH/line layout and units, not the number of emission lines.
SCHEMA = 'cloudy_hdf5_mvp_v1'


def read_cloudy_hdf5_table(filename) -> dict:
    """Read the project's Cloudy axes, labels, and linear volume emissivities.

    The v1 layout supports any positive number of lines. Both the original
    single-line table and the multi-line converter use the same schema tag.
    The tag alone is not sufficient: datasets, units, axis order, and shape
    are checked as well. This is a project format, not a universal Cloudy one.

    Returns: table (dict): axes, labels, fixed-abundance description, and
        emissivity array ordered as (temperature, density, line).
    """
    with h5py.File(filename, 'r') as source:
        schema = source.attrs.get('schema')
        if schema != SCHEMA:
            raise ValueError(
                f'{filename}: unsupported HDF5 schema {schema!r}; expected {SCHEMA!r}. '
                'This identifier supports both single-line and multi-line tables.'
            )
        required = ('axes/log_temperature_K', 'axes/log_nH_cm3',
                    'lines/labels', 'lines/emissivity_erg_s_cm3')
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f'{filename}: Cloudy HDF5 table is missing required datasets: {missing}.')
        temperature = np.asarray(source['axes/log_temperature_K'], dtype=np.float64)
        density = np.asarray(source['axes/log_nH_cm3'], dtype=np.float64)
        labels = list(source['lines/labels'].asstr()[...])
        values = np.asarray(source['lines/emissivity_erg_s_cm3'], dtype=np.float64)
        dataset = source['lines/emissivity_erg_s_cm3']
        if (dataset.attrs.get('unit') != 'erg s^-1 cm^-3'
                or dataset.attrs.get('axis_order') != 'log_temperature_K, log_nH_cm3, line'):
            raise ValueError('Cloudy HDF5 emissivity unit or axis order is unsupported.')
        abundance_model = str(source.attrs['abundance_model'])

    for name, axis in (('temperature', temperature), ('density', density)):
        if (axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis))
                or not np.all(np.diff(axis) > 0)):
            raise ValueError(f'{name} axis must be finite and strictly increasing.')
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Cloudy HDF5 line labels must be nonempty and unique.')
    if values.shape != (temperature.size, density.size, len(labels)):
        raise ValueError('Cloudy HDF5 emissivity shape does not match its axes.')
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('Log interpolation requires finite positive emissivities.')
    return {
        'axes': {'log_temperature_K': temperature, 'log_nH_cm3': density},
        'line_labels': labels,
        'volume_emissivity_erg_s_cm3': values,
        'abundance_model': abundance_model,
    }


def calculate_particle_line_luminosity(gas_fields: dict, *, table_filename: str,
                                       redshift: float, line_label: str) -> dict:
    """Calculate one line through the same interpolation used for multi-line runs.

    Returns: emission (dict): emissivity, luminosity, abundance and clipping data.
    """
    table = read_cloudy_hdf5_table(table_filename)
    return next(iter_particle_line_luminosities(
        gas_fields, table=table, redshift=redshift, line_labels=[line_label],
    ))


def iter_particle_line_luminosities(gas_fields: dict, *, table: dict,
                                    redshift: float, line_labels: list):
    """Reuse one loaded Cloudy table and gas interpolation points for all lines.

    Finite positive T/nH values outside the grid are clipped for interpolation
    only. One warning per affected axis reports particle counts, not line counts.
    Original gas values and particle volumes are unchanged. The table has fixed
    abundances and no metallicity or redshift axis; it does not rescale to SIMBA.

    Returns: emissions (collections.abc.Iterator): one emission dictionary per
        requested label, in input order, without retaining all particle arrays.
    """
    if not np.isfinite(float(redshift)):
        raise ValueError('redshift must be finite.')
    if (not isinstance(line_labels, list) or not line_labels
            or any(not isinstance(label, str) or not label.strip() for label in line_labels)):
        raise ValueError('line_labels must be a nonempty list of exact identifiers.')
    if len(set(line_labels)) != len(line_labels):
        raise ValueError('Repeated line labels requested.')
    missing = [label for label in line_labels if label not in table['line_labels']]
    if missing:
        raise ValueError(f'Lines {missing!r} absent; available: {table["line_labels"]}')
    fields = {}
    for name in ('nH_cm3', 'T_K', 'volume_cm3'):
        if name not in gas_fields:
            raise KeyError(f'Missing Cloudy MVP gas field: {name}')
        values = np.asarray(gas_fields[name], dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f'{name} must be a finite, positive, one-dimensional array.')
        fields[name] = values
    if len({values.size for values in fields.values()}) != 1:
        raise ValueError('Cloudy MVP fields must align by particle.')

    log_temperature = np.log10(fields['T_K'])
    log_density = np.log10(fields['nH_cm3'])
    axes = table['axes']
    clipped = np.zeros(log_temperature.size, dtype=bool)
    for name, data, axis in (
        ('log_temperature_K', log_temperature, axes['log_temperature_K']),
        ('log_nH_cm3', log_density, axes['log_nH_cm3']),
    ):
        below = data < axis[0]
        above = data > axis[-1]
        clipped |= below | above
        if np.any(below) or np.any(above):
            warnings.warn(
                f'Cloudy HDF5 {name} clipping: {np.count_nonzero(below)} below, '
                f'{np.count_nonzero(above)} above the table range '
                f'[{axis[0]:g}, {axis[-1]:g}]. Using boundary values for '
                'interpolation only; original gas fields are unchanged.',
                RuntimeWarning, stacklevel=2,
            )
    log_temperature = np.clip(log_temperature, *axes['log_temperature_K'][[0, -1]])
    log_density = np.clip(log_density, *axes['log_nH_cm3'][[0, -1]])
    points = np.column_stack((log_temperature, log_density))
    warnings.warn(
        'Cloudy MVP uses fixed Cloudy default abundances; SIMBA oxygen and total '
        'metallicity do not rescale this emissivity.', RuntimeWarning, stacklevel=2,
    )
    for label in line_labels:
        index = table['line_labels'].index(label)
        interpolator = RegularGridInterpolator(
            (axes['log_temperature_K'], axes['log_nH_cm3']),
            np.log10(table['volume_emissivity_erg_s_cm3'][:, :, index]),
            bounds_error=True,
        )
        log_emissivity = interpolator(points)
        volume_emissivity = 10.0 ** log_emissivity
        yield {
            'line_label': label,
            'log_volume_emissivity': log_emissivity,
            'volume_emissivity_erg_s_cm3': volume_emissivity,
            'line_luminosity_erg_s': volume_emissivity * fields['volume_cm3'],
            'abundance_model': table['abundance_model'],
            'clipped_particle_count': int(np.count_nonzero(clipped)),
        }
