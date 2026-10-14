"""Input-source adapters used by the line-emission controller."""

import h5py
import numpy as np

from .simba import (
    convert_gas_fields,
    get_component_by_particle_ids,
    select_host_halo_particle_ids,
)


SIMBA_GAS_COMPONENTS = [
    'PartType0/ParticleIDs',
    'PartType0/Coordinates',
    'PartType0/Velocities',
    'PartType0/Masses',
    'PartType0/Density',
    'PartType0/InternalEnergy',
    'PartType0/Metallicity',
    'PartType0/ElectronAbundance',
]


def select_gas_particle_ids(source_type: str, snapshot_filename: str, selection: dict):
    """Return ParticleIDs from a saved ID array or a source-specific selector."""
    if selection['type'] == 'particle_ids_file':
        return np.load(selection['filename'], allow_pickle=False)
    if source_type.lower() == 'simba' and selection['type'] == 'caesar_host_halo':
        return select_host_halo_particle_ids(
            snapshot_filename,
            selection['catalogue_filename'],
            int(selection['target_galaxy_idx']),
        )
    raise ValueError(f"Unknown selection for {source_type}: {selection['type']}")


def load_physical_gas_fields(
    source_type: str,
    snapshot_filename: str,
    particle_ids,
    unit_factors: dict,
) -> dict:
    """Read selected particles and return the shared physical-field layout."""
    if source_type.lower() == 'simba':
        raw_fields = get_component_by_particle_ids(
            snapshot_filename,
            SIMBA_GAS_COMPONENTS,
            particle_ids,
        )
        # Snapshot floats can overflow when physical unit factors are large.
        for name, values in raw_fields.items():
            if name != 'PartType0/ParticleIDs':
                raw_fields[name] = values.astype(np.float64, copy=False)
        return convert_gas_fields(raw_fields, **unit_factors)
    raise ValueError(f'Unknown input source: {source_type}')


def get_source_redshift(source_type: str, snapshot_filename: str) -> float:
    """Read redshift from the selected source's snapshot metadata."""
    if source_type.lower() == 'simba':
        with h5py.File(snapshot_filename, 'r') as snapshot:
            return float(snapshot['Header'].attrs['Redshift'])
    raise ValueError(f'Unknown input source: {source_type}')
