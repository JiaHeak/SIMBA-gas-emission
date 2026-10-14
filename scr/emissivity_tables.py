# Adapters for precomputed gas-line emissivity tables.
import numpy as np
from .tables import ps20

def calculate_ps20_particle_line_luminosity(gas_fields: dict, *, table_filename: str, redshift: float, line_label: str) -> dict:
    """Calculate PS20 emissivity and line luminosity for each gas particle.

    ``gas_fields`` is the physical-quantity dictionary produced by
    ``simba.convert_gas_fields``. PS20 is interpolated using hydrogen number
    density, temperature, and metallicity; the resulting volume emissivity is
    multiplied by each particle's volume to get its line luminosity.

    The returned luminosity is in erg s^-1, assuming the PS20 table values
    are log10(4 pi J_line) in erg cm^-3 s^-1 and ``volume_cm3`` is in cm^3.

    Returns: emission (dict): emissivity and line-luminosity arrays per particle.
    """
    required_fields = ('nH_cm3', 'T_K', 'Z_Zsun', 'volume_cm3')
    missing_fields = [name for name in required_fields if name not in gas_fields]
    if missing_fields:
        raise KeyError(f'Missing gas fields required for emissivity: {missing_fields}')

    particle_fields = {
        name: np.asarray(gas_fields[name], dtype=np.float64)
        for name in required_fields
    }
    for name, values in particle_fields.items():
        if values.ndim != 1:
            raise ValueError(f'{name} must be a one-dimensional particle array.')
        if not np.all(np.isfinite(values)):
            raise ValueError(f'{name} contains non-finite values.')
        if np.any(values <= 0):
            raise ValueError(f'{name} must contain only positive values.')

    field_lengths = {values.size for values in particle_fields.values()}
    if len(field_lengths) != 1:
        raise ValueError('Emissivity input fields must have matching particle counts.')

    log_nH = np.log10(particle_fields['nH_cm3'])
    log_T = np.log10(particle_fields['T_K'])
    log_Z = np.log10(particle_fields['Z_Zsun'])

    log_volume_emissivity = ps20.interpolate_log_emissivity(
        filename_table=table_filename,
        log_nH=log_nH,
        log_T=log_T,
        log_Z=log_Z,
        redshift=redshift,
        line_label=line_label,
    )
    volume_emissivity = 10.0 ** log_volume_emissivity
    particle_luminosity = volume_emissivity * particle_fields['volume_cm3']

    return {
        'log_volume_emissivity': log_volume_emissivity,
        'volume_emissivity_erg_s_cm3': volume_emissivity,
        'line_luminosity_erg_s': particle_luminosity,
    }


def calculate_particle_line_luminosity(gas_fields: dict, *, table_type: str, table_filename: str, redshift: float, line_label: str) -> dict:
    """Dispatch to the selected emissivity-table adapter.

    Each adapter returns ``line_luminosity_erg_s`` for the cube maker.
    Register future table adapters here; the main pipeline need not change.

    Returns: emission (dict): emissivity and line-luminosity arrays per particle.
    """
    adapters = {
        'ps20': calculate_ps20_particle_line_luminosity,
    }
    table_name = table_type.lower()
    if table_name not in adapters:
        raise ValueError(
            f'Unknown emissivity table type {table_type!r}; '
            f'available types: {sorted(adapters)}'
        )
    adapter = adapters[table_name]
    return adapter(
        gas_fields,
        table_filename=table_filename,
        redshift=redshift,
        line_label=line_label,
    )
