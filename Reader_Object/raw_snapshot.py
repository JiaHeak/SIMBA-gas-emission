"""Read physical gas fields on demand from an original SIMBA snapshot."""
from pathlib import Path

import h5py
import numpy as np

from scr.simba import compute_unit_factors, gas_temperature_and_hydrogen_fraction, get_gas_fields


# Same SIMBA conventions and constants as config.example.yaml. Override with
# the pipeline's unit_inputs when a run uses different constants or gamma.
DEFAULT_UNIT_INPUTS = dict(solar_mass_g=1.988416e33, kpc_to_cm=3.0856775814913673e21,
                         specific_energy_to_erg_g=1.e10, z_sun_mass_fraction=.0134,
                         gamma=5. / 3.)


class RawSnapshot:
    """A read-only SIMBA snapshot with lazy particle access."""

    def __init__(self, filename, *, unit_inputs=None):
        """Read the small header and remember the original file path.

        Returns: result (NoneType): initializes the snapshot reader.
        """
        self.filename = Path(filename).expanduser().resolve()
        self.unit_inputs = dict(DEFAULT_UNIT_INPUTS if unit_inputs is None else unit_inputs)
        with h5py.File(self.filename, 'r') as source:
            header = source['Header'].attrs
            self.scale_factor = float(header['Time'])
            self.hubble_param = float(header['HubbleParam'])
        self.unit_factors = compute_unit_factors(str(self.filename), self.unit_inputs)

    def read_gas_fields(self, rows, *, fields, unit_factors=None, particle_ids=None):
        """Read physical gas fields for selected rows, preserving their order.

        Defaults to this snapshot's factors. Explicit factors support the
        existing catalogue interface. Positions are absolute physical kpc,
        velocities are peculiar km/s in the simulation frame, and density
        is physical g/cm³. Supply
        particle_ids to check row alignment.

        Returns: values (dict[str, numpy.ndarray]): fields in requested row order.
        """
        fields = {fields} if isinstance(fields, str) else set(fields)
        if not fields or fields - {'mass_g', 'T_K', 'rho_g_cm3', 'position_kpc', 'velocity_kms'}:
            raise ValueError('Select mass_g, T_K, rho_g_cm3, position_kpc, or velocity_kms.')
        unit_factors = self.unit_factors if unit_factors is None else unit_factors
        for key, expected in [('scale_factor', self.scale_factor), ('hubble_param', self.hubble_param)]:
            if key in unit_factors and not np.isclose(unit_factors[key], expected, rtol=1.e-10, atol=0):
                raise ValueError(f'Unit factors have the wrong {key} for this snapshot.')
        components = []
        if 'mass_g' in fields:
            components.append('PartType0/Masses')
        if 'rho_g_cm3' in fields:
            components.append('PartType0/Density')
        if 'T_K' in fields:
            components.extend(['PartType0/InternalEnergy', 'PartType0/Metallicity',
                               'PartType0/ElectronAbundance'])
        for field, dataset in [('position_kpc', 'Coordinates'), ('velocity_kms', 'Velocities')]:
            if field in fields:
                components.append('PartType0/' + dataset)
        raw = get_gas_fields(self.filename, rows, components)
        if particle_ids is not None and not np.array_equal(raw['PartType0/ParticleIDs'], particle_ids):
            raise ValueError('Snapshot ParticleIDs do not match the saved catalogue rows.')
        values = {}
        if 'mass_g' in fields:
            values['mass_g'] = np.asarray(raw['PartType0/Masses'], dtype=float) * unit_factors['mass_to_g']
        if 'rho_g_cm3' in fields:
            values['rho_g_cm3'] = np.asarray(raw['PartType0/Density'], dtype=float) * unit_factors['density_to_g_cm3']
        if 'T_K' in fields:
            values['T_K'], _ = gas_temperature_and_hydrogen_fraction(
                raw, specific_energy_to_erg_g=unit_factors['specific_energy_to_erg_g'],
                gamma=unit_factors['gamma'])
        for field, dataset, factor in [('position_kpc', 'Coordinates', 'position_to_physical_kpc'),
                                       ('velocity_kms', 'Velocities', 'velocity_to_kms')]:
            if field in fields:
                values[field] = np.asarray(raw['PartType0/' + dataset], dtype=float) * unit_factors[factor]
                if values[field].shape != (len(rows), 3):
                    raise ValueError(f'{dataset} must have shape (N, 3).')
        for field, value in values.items():
            if not np.all(np.isfinite(value)) or (field in {'mass_g', 'T_K', 'rho_g_cm3'} and np.any(value < 0)):
                raise ValueError(f'Invalid physical values in {field}.')
        return values


def read_RawSnapshot(filename, *, unit_inputs=None):
    """Open an original SIMBA snapshot for lazy gas access.

    Returns: snapshot (RawSnapshot): reader with the source path and cosmology.
    """
    return RawSnapshot(filename, unit_inputs=unit_inputs)
