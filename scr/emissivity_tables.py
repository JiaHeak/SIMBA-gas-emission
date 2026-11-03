"""Dispatch physical gas fields to the selected emissivity-table module."""

from pathlib import Path

from .tables import cloudy_hdf5, ps20


def calculate_particle_line_luminosity(gas_fields: dict, *, table_type: str,
                                       table_filename: str, redshift: float, line_label: str) -> dict:
    """Validate the selected model and file, then dispatch the calculation.

    table_type selects ps20 or cloudy_hdf5. Both adapters read a single HDF5
    table; raw Cloudy text conversion is separate from cube production.

    Returns: emission (dict): per-particle emissivity and luminosity arrays.
    """
    if not isinstance(table_type, str) or not table_type:
        raise ValueError('table_type must be a nonempty model name, such as ps20.')
    adapters = {
        'ps20': ps20.calculate_particle_line_luminosity,
        'cloudy_hdf5': cloudy_hdf5.calculate_particle_line_luminosity,
    }
    table_name = table_type.lower()
    if table_name not in adapters:
        raise ValueError(
            f'Unknown emissivity table type {table_type!r}; '
            f'available types: {sorted(adapters)}'
        )
    if not isinstance(table_filename, (str, Path)):
        raise TypeError('table_filename must be a file path (str or pathlib.Path).')
    assert Path(table_filename).is_file(), f'Emissivity table file not found: {table_filename}'
    return adapters[table_name](
        gas_fields, table_filename=table_filename,
        redshift=redshift, line_label=line_label,
    )


def load_emissivity_table(table_type: str, table_filename: str, line_labels: list) -> dict:
    """Load model metadata and validate all requested lines before gas selection.

    Cloudy stores its small complete table here for reuse across lines. PS20
    keeps its existing per-line reader, avoiding a full large-table load.

    Returns: emissivity_table (dict): model, path, requested labels, and table data.
    """
    if table_type not in ('cloudy_hdf5', 'ps20'):
        raise ValueError('table_type must be cloudy_hdf5 or ps20.')
    if (not isinstance(line_labels, list) or not line_labels
            or any(not isinstance(label, str) or not label.strip() for label in line_labels)):
        raise ValueError('line_labels must be a nonempty list of exact identifiers.')
    if len(set(line_labels)) != len(line_labels):
        raise ValueError('Repeated line labels requested.')
    if not Path(table_filename).is_file():
        raise FileNotFoundError(f'Emissivity table file not found: {table_filename}')
    reader = cloudy_hdf5.read_cloudy_hdf5_table if table_type == 'cloudy_hdf5' else ps20.get_ps20_metadata
    data = reader(table_filename)
    missing = [label for label in line_labels if label not in data['line_labels']]
    if missing:
        raise ValueError(f'Requested lines absent: {missing}; available: {data["line_labels"]}')
    return {'table_type': table_type, 'table_filename': str(table_filename),
            'line_labels': list(line_labels), 'data': data}


def iter_particle_line_luminosities(gas_fields: dict, *, emissivity_table: dict, redshift: float):
    """Dispatch an ordered batch without rereading the Cloudy table per line.

    Returns: emissions (collections.abc.Iterator): per-line particle luminosities
        and adapter diagnostics; the shared gas fields are not modified.
    """
    if emissivity_table['table_type'] == 'cloudy_hdf5':
        yield from cloudy_hdf5.iter_particle_line_luminosities(
            gas_fields, table=emissivity_table['data'], redshift=redshift,
            line_labels=emissivity_table['line_labels'],
        )
    elif emissivity_table['table_type'] == 'ps20':
        for label in emissivity_table['line_labels']:
            emission = ps20.calculate_particle_line_luminosity(
                gas_fields, table_filename=emissivity_table['table_filename'],
                redshift=redshift, line_label=label,
            )
            yield {'line_label': label, **emission}
    else:
        raise ValueError('Unsupported prepared emissivity table.')

