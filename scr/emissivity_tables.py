"""Dispatch physical gas fields to the selected emissivity-table module."""

from pathlib import Path

from .tables import ps20


def calculate_particle_line_luminosity(gas_fields: dict, *, table_type: str,
                                       table_filename: str, redshift: float, line_label: str) -> dict:
    """Calculate line luminosity using the configured table implementation.

    Add future table modules to ``adapters`` without changing the controller.

    Returns: emission (dict): per-particle emissivity and luminosity arrays.
    """
    adapters = {
        'ps20': ps20.calculate_particle_line_luminosity,
    }
    table_name = table_type.lower()
    if table_name not in adapters:          # Hey Codex, check this function and how it is called in Pipeline
        raise ValueError(
            f'Unknown emissivity table type {table_type!r}; '
            f'available types: {sorted(adapters)}')

    assert Path(table_filename).is_file(), f'Emissivity table file not found: {table_filename}'

    return adapters[table_name](
        gas_fields,
        table_filename=table_filename,
        redshift=redshift,
        line_label=line_label,
    )
