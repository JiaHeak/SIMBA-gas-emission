"""Dispatch physical gas fields to the selected emissivity-table module."""

from pathlib import Path

from .tables import cloudy_hdf5, cloudy_mvp, ps20


def calculate_particle_line_luminosity(gas_fields: dict, *, table_type: str,
                                       table_filename: str, redshift: float, line_label: str,
                                       grid_filename=None) -> dict:
    """Validate the selected model and file, then dispatch the calculation.

    table_type selects ps20, cloudy_mvp, or cloudy_hdf5. Only the text-pair
    cloudy_mvp adapter requires grid_filename; HDF5 adapters use one file.
    The previous dispatcher is in emissivity_tables_hdf5_old.py.

    Returns: emission (dict): per-particle emissivity and luminosity arrays.
    """
    if not isinstance(table_type, str) or not table_type:
        raise ValueError('table_type must be a nonempty model name, such as ps20.')
    adapters = {
        'ps20': ps20.calculate_particle_line_luminosity,
        'cloudy_mvp': cloudy_mvp.calculate_particle_line_luminosity,
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
    table_options = {}
    if table_name == 'cloudy_mvp':
        if grid_filename is None:
            raise ValueError('cloudy_mvp requires an explicit grid_filename alongside table_filename (.emis).')
        if not isinstance(grid_filename, (str, Path)):
            raise TypeError('grid_filename must be a file path (str or pathlib.Path).')
        assert Path(grid_filename).is_file(), f'Cloudy grid file not found: {grid_filename}'
        table_options['grid_filename'] = grid_filename
    elif grid_filename is not None:
        raise ValueError('grid_filename is only supported by the cloudy_mvp adapter.')
    return adapters[table_name](
        gas_fields, table_filename=table_filename,
        redshift=redshift, line_label=line_label, **table_options,
    )
