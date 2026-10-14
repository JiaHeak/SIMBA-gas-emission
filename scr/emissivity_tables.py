# Here are the pre-computed emissivity tables for the gas line emissions. 
import h5py as h5
import numpy as np
from .tables.ps20 import interpolate_log_emissivity, decode_labels


# Original draft order was ``table='ps20', filename: str``. Python requires a
# required parameter after a default parameter to be keyword-only.

def calculate_ps20_particle_line_luminosity(gas_fields: dict, *, table_filename: str, redshift: float, line_label: str) -> dict:
    """Calculate PS20 emissivity and line luminosity for each gas particle.

    ``gas_fields`` is the physical-quantity dictionary produced by
    ``convert_gas_fields_for_review``. PS20 is interpolated using hydrogen
    number density, temperature, and metallicity; the resulting volume
    emissivity is multiplied by each particle's volume to get its line
    luminosity.

    The returned luminosity is in erg s^-1, assuming the PS20 table values
    are log10(4 pi J_line) in erg cm^-3 s^-1 and ``volume_cm3`` is in cm^3.
    """
    log_nH = np.log10(gas_fields['nH_cm3'])
    log_T = np.log10(gas_fields['T_K'])
    log_Z = np.log10(gas_fields['Z_Zsun'])

    log_volume_emissivity = interpolate_log_emissivity(
        filename_table=table_filename,
        log_nH=log_nH,
        log_T=log_T,
        log_Z=log_Z,
        redshift=redshift,
        line_label=line_label,
    )
    volume_emissivity = 10.0 ** log_volume_emissivity
    particle_luminosity = volume_emissivity * gas_fields['volume_cm3']

    return {
        'log_volume_emissivity': log_volume_emissivity,
        'volume_emissivity_erg_s_cm3': volume_emissivity,
        'line_luminosity_erg_s': particle_luminosity,
    }


def calculate_particle_line_luminosity(
    gas_fields: dict,
    *,
    table_type: str,
    table_filename: str,
    redshift: float,
    line_label: str,
) -> dict:
    """Dispatch to the selected emissivity-table adapter.

    Each adapter returns ``line_luminosity_erg_s`` for the cube maker.
    Register future table adapters here; the main pipeline need not change.
    """
    adapters = {
        'ps20': calculate_ps20_particle_line_luminosity,
    }
    adapter = adapters[table_type.lower()]
    return adapter(
        gas_fields,
        table_filename=table_filename,
        redshift=redshift,
        line_label=line_label,
    )
