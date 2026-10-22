# Read SIMBA gas using snapshot rows selected by catalogue membership.
from pathlib import Path

import h5py
import numpy as np

SIMBA_GAS_COMPONENTS = [
    'PartType0/ParticleIDs',
    'PartType0/Coordinates',
    'PartType0/Velocities',
    'PartType0/Masses',
    'PartType0/Density',
    'PartType0/InternalEnergy',
    'PartType0/Metallicity',
    'PartType0/ElectronAbundance']


def select_gas_rows(catalogue_filename: str, target_index: int,
                    group_type: str = 'halo') -> np.ndarray:
    """Select snapshot gas rows from an Octavius halo or galaxy.

    Returns: gas_rows (numpy.ndarray): unique integer rows in the original snapshot.
    """
    from octavius import load_catalogue

    if group_type not in {'halo', 'galaxy'}:
        raise ValueError('group_type must be halo or galaxy.')
    if (
        isinstance(target_index, (bool, np.bool_))
        or not isinstance(target_index, (int, np.integer))
        or target_index < 0
    ):
        raise ValueError('target_index must be a nonnegative integer.')
    with load_catalogue(Path(catalogue_filename)) as catalogue:
        groups = catalogue.haloes if group_type == 'halo' else catalogue.galaxies
        if groups is None or target_index >= len(groups):
            raise IndexError(f'target_index is outside the Octavius {group_type} catalogue.')
        gas_rows = np.asarray(groups.get_particle_indices(
            group_index=target_index, ptype='gas',
        ))
    if gas_rows.ndim != 1 or not np.issubdtype(gas_rows.dtype, np.integer):
        raise ValueError('Octavius gas indices must be a one-dimensional integer array.')
    gas_rows = gas_rows.astype(np.int64, copy=False)
    assert np.unique(gas_rows).size == gas_rows.size, 'Repeated Octavius gas rows.'
    return gas_rows


def get_reference_frame(catalogue_filename: str, target_index: int,
                        group_type: str = 'halo') -> dict:
    """Read an object's physical center and bulk velocity from Octavius.

    Halo centers use minpot_pos and velocities use com_vel_total. Galaxy
    centers and velocities use the baryonic center-of-mass quantities.
    Catalogue unit conversions supply physical kpc and km/s directly.

    Returns: reference_frame (dict): center_kpc and bulk_velocity_kms 3-vectors.
    """
    from octavius import load_catalogue

    if group_type not in {'halo', 'galaxy'}:
        raise ValueError('group_type must be halo or galaxy.')
    if (
        isinstance(target_index, (bool, np.bool_))
        or not isinstance(target_index, (int, np.integer))
        or target_index < 0
    ):
        raise ValueError('target_index must be a nonnegative integer.')
    position_name = 'minpot_pos' if group_type == 'halo' else 'com_pos_baryon'
    velocity_name = 'com_vel_total' if group_type == 'halo' else 'com_vel_baryon'
    with load_catalogue(Path(catalogue_filename)) as catalogue:
        groups = catalogue.haloes if group_type == 'halo' else catalogue.galaxies
        if groups is None or target_index >= len(groups):
            raise IndexError(f'target_index is outside the Octavius {group_type} catalogue.')
        mask = np.array([target_index], dtype=np.int64)
        center = np.asarray(groups.get_dataset(
            position_name, mask=mask, to_units='kpc', to_physical=True,
        ), dtype=np.float64)[0]
        velocity = np.asarray(groups.get_dataset(
            velocity_name, mask=mask, to_units='km/s', to_physical=True,
        ), dtype=np.float64)[0]
    if center.shape != (3,) or velocity.shape != (3,):
        raise ValueError('Octavius center and bulk velocity must be 3-vectors.')
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(velocity)):
        raise ValueError('Octavius reference frame contains non-finite values.')
    return {'center_kpc': center, 'bulk_velocity_kms': velocity}


def get_catalogue_redshift(catalogue_filename: str) -> float:
    """Read redshift from an Octavius catalogue.

    Returns: redshift (float): catalogue snapshot redshift.
    """
    from octavius import load_catalogue

    with load_catalogue(Path(catalogue_filename)) as catalogue:
        return float(catalogue.redshift)


def get_gas_fields(filename, gas_rows, components=SIMBA_GAS_COMPONENTS) -> dict:
    """Read snapshot gas rows, preserving requested order and particle alignment.

    Returns: gas_fields (dict): NumPy arrays keyed by HDF5 path.
    """
    rows = np.asarray(gas_rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError('gas_rows must be a one-dimensional integer array.')
    rows = rows.astype(np.int64, copy=False)
    assert np.unique(rows).size == rows.size, 'Repeated gas rows requested.'

    with h5py.File(filename, 'r') as snapshot:
        gas_count = snapshot['PartType0/ParticleIDs'].shape[0]
        if np.any(rows < 0) or np.any(rows >= gas_count):
            raise IndexError('A requested gas row is outside the snapshot.')

        sorted_order = np.argsort(rows)
        sorted_rows = rows[sorted_order]
        restore_order = np.argsort(sorted_order)
        requested_components = list(components)
        if 'PartType0/ParticleIDs' not in requested_components:
            requested_components.append('PartType0/ParticleIDs')

        return {
            component: snapshot[component][sorted_rows][restore_order]
            if rows.size else snapshot[component][:0]
            for component in requested_components
        }


def convert_gas_fields(                 # This is given by Codex, we should make sure the unit convention !!!
    raw_fields: dict, *, position_to_physical_kpc: float, velocity_to_kms: float, mass_to_g: float,
    density_to_g_cm3: float, specific_energy_to_erg_g: float, z_sun_mass_fraction: float, gamma: float = 5.0 / 3.0 ) -> dict:
    """Convert selected SIMBA arrays to the physical fields used downstream.

    Conversion factors must already account for the snapshot's scale factor
    and Hubble convention. The abundance columns and electron abundance are
    interpreted according to the SIMBA snapshot convention.
    """
    proton_mass_g = 1.67262192369e-24
    boltzmann_erg_k = 1.380649e-16

    coordinates = np.asarray(
        raw_fields['PartType0/Coordinates'], dtype=np.float64
    ) * position_to_physical_kpc
    velocities = np.asarray(
        raw_fields['PartType0/Velocities'], dtype=np.float64
    ) * velocity_to_kms
    mass_g = np.asarray(raw_fields['PartType0/Masses'], dtype=np.float64) * mass_to_g
    rho_g_cm3 = np.asarray(
        raw_fields['PartType0/Density'], dtype=np.float64
    ) * density_to_g_cm3
    internal_energy_erg_g = np.asarray(
        raw_fields['PartType0/InternalEnergy'], dtype=np.float64
    ) * specific_energy_to_erg_g
    abundances = np.asarray(raw_fields['PartType0/Metallicity'], dtype=np.float64)
    electron_abundance = np.asarray(
        raw_fields['PartType0/ElectronAbundance'], dtype=np.float64
    )

    z_mass_fraction = abundances[:, 0]
    helium_mass_fraction = abundances[:, 1]
    hydrogen_mass_fraction = 1.0 - z_mass_fraction - helium_mass_fraction
    helium_to_hydrogen_number = helium_mass_fraction / (4.0 * hydrogen_mass_fraction)
    mean_molecular_weight = 1.0 / (
        hydrogen_mass_fraction
        * (1.0 + helium_to_hydrogen_number + electron_abundance)
    )
    temperature_k = (
        (gamma - 1.0)
        * internal_energy_erg_g
        * mean_molecular_weight
        * proton_mass_g
        / boltzmann_erg_k
    )

    return {
        'particle_id': raw_fields['PartType0/ParticleIDs'],
        'x_kpc': coordinates[:, 0],
        'y_kpc': coordinates[:, 1],
        'z_kpc': coordinates[:, 2],
        'v_x_kms': velocities[:, 0],
        'v_y_kms': velocities[:, 1],
        'v_z_kms': velocities[:, 2],
        'mass_g': mass_g,
        'rho_g_cm3': rho_g_cm3,
        'volume_cm3': mass_g / rho_g_cm3,
        'nH_cm3': hydrogen_mass_fraction * rho_g_cm3 / proton_mass_g,
        'T_K': temperature_k,
        'Z_Zsun': z_mass_fraction / z_sun_mass_fraction,
        'C_mass_fraction': abundances[:, 2],
        'O_mass_fraction': abundances[:, 4],
    }


# Physical-field loader used by the shared pipeline after gas rows are selected.
def load_physical_gas_fields(snapshot_filename: str, gas_rows, unit_factors: dict) -> dict:
    """Read selected snapshot gas rows and convert them to physical fields.

    Returns: gas_fields (dict): aligned physical arrays, including ParticleIDs.
    """
    raw_fields = get_gas_fields(snapshot_filename, gas_rows, SIMBA_GAS_COMPONENTS)
    return convert_gas_fields(raw_fields, **unit_factors)


def get_redshift(snapshot_filename: str) -> float:
    """Read the SIMBA snapshot redshift.

    Returns: redshift (float): value stored in Header/Redshift.
    """
    with h5py.File(snapshot_filename, 'r') as snapshot:
        return float(snapshot['Header'].attrs['Redshift'])
