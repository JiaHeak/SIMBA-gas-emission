"""Read physical gas fields on demand from an original SIMBA snapshot."""
from pathlib import Path

import h5py
import numpy as np

from scr.simba import gas_temperature_and_hydrogen_fraction, get_gas_fields


class RawSnapshot:
    """A read-only SIMBA snapshot with lazy particle access."""

    def __init__(self, filename):
        """Read the small header and remember the original file path.

        Returns: result (NoneType): initializes the snapshot reader.
        """
        self.filename = Path(filename).expanduser().resolve()
        with h5py.File(self.filename, 'r') as source:
            header = source['Header'].attrs
            self.scale_factor = float(header['Time'])
            self.hubble_param = float(header['HubbleParam'])

    def read_gas_fields(self, rows, *, fields, unit_factors, particle_ids=None):
        """Read mass in g and/or derive temperature in K for selected gas rows.

        Use physical conversion factors from the emission catalogue or from
        the SIMBA pipeline. Supply particle_ids to check the row alignment.

        Returns: values (dict[str, numpy.ndarray]): fields in requested row order.
        """
        fields = set(fields)
        if not fields or fields - {'mass_g', 'T_K'}:
            raise ValueError("Select 'mass_g', 'T_K', or both.")
        components = []
        if 'mass_g' in fields:
            components.append('PartType0/Masses')
        if 'T_K' in fields:
            components.extend(['PartType0/InternalEnergy', 'PartType0/Metallicity',
                               'PartType0/ElectronAbundance'])
        raw = get_gas_fields(self.filename, rows, components)
        if particle_ids is not None and not np.array_equal(raw['PartType0/ParticleIDs'], particle_ids):
            raise ValueError('Snapshot ParticleIDs do not match the saved catalogue rows.')
        values = {}
        if 'mass_g' in fields:
            values['mass_g'] = np.asarray(raw['PartType0/Masses'], dtype=float) * unit_factors['mass_to_g']
        if 'T_K' in fields:
            values['T_K'], _ = gas_temperature_and_hydrogen_fraction(
                raw, specific_energy_to_erg_g=unit_factors['specific_energy_to_erg_g'],
                gamma=unit_factors['gamma'])
        return values


def read_RawSnapshot(filename):
    """Open an original SIMBA snapshot for lazy gas access.

    Returns: snapshot (RawSnapshot): reader with the source path and cosmology.
    """
    return RawSnapshot(filename)
