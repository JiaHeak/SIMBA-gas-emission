# Read and convert gas data from SIMBA snapshots and CAESAR catalogues.
import h5py
import numpy as np
from caesar import load

SIMBA_GAS_COMPONENTS = [
    'PartType0/ParticleIDs',
    'PartType0/Coordinates',
    'PartType0/Velocities',
    'PartType0/Masses',
    'PartType0/Density',
    'PartType0/InternalEnergy',
    'PartType0/Metallicity',
    'PartType0/ElectronAbundance']


def select_halo_particle_ids(snapshot_filename: str, catalogue_filename: str, target_galaxy_idx: int, find_host: bool = False) -> np.ndarray:
    """Convert CAESAR gas rows to ParticleIDs in the catalogue's order.

    Returns: particle_ids (numpy.ndarray): unique selected IDs in catalogue order.
    """
    catalogue = load(str(catalogue_filename))
    galaxy = catalogue.galaxies[target_galaxy_idx]
    gas_rows = np.asarray(
        galaxy.halo.glist if find_host else galaxy.glist,
        dtype=np.int64,
    )
    assert np.unique(gas_rows).size == gas_rows.size, (
        'Repeated gas rows found in the CAESAR selection.'
    )

    with h5py.File(snapshot_filename, 'r') as snapshot:
        ids = snapshot['PartType0/ParticleIDs']
        if gas_rows.size == 0:
            return ids[:0]
        sorted_order = np.argsort(gas_rows)
        sorted_rows = gas_rows[sorted_order]
        selected_ids = ids[sorted_rows]
        restore_order = np.argsort(sorted_order)
        selected_ids = selected_ids[restore_order]
        assert np.unique(selected_ids).size == selected_ids.size, (
            'Repeated ParticleIDs found in the CAESAR selection.'
        )
        return selected_ids


def get_gas_fields_by_id(filename, particle_ids, components=SIMBA_GAS_COMPONENTS) -> dict:
    """Read selected gas fields, preserving ParticleID order and vector shapes.

    Components use full HDF5 paths such as ``PartType0/Coordinates``.
    Returns: gas_fields (dict): NumPy arrays keyed by HDF5 path.
    """
    with h5py.File(filename, "r") as snapshot:
        snapshot_ids = snapshot["PartType0/ParticleIDs"][...]
        requested_ids = np.asarray(particle_ids, dtype=snapshot_ids.dtype).ravel()
        assert np.unique(snapshot_ids).size == snapshot_ids.size, (
            'Repeated ParticleIDs found in the snapshot.'
        )
        assert np.unique(requested_ids).size == requested_ids.size, (
            'Repeated IDs found in the requested ParticleID list.'
        )

        if requested_ids.size:
            sorted_rows = np.argsort(snapshot_ids)
            sorted_ids = snapshot_ids[sorted_rows]
            positions = np.searchsorted(sorted_ids, requested_ids)
            found = positions < sorted_ids.size
            found[found] &= sorted_ids[positions[found]] == requested_ids[found]
            if not np.all(found):
                raise KeyError(f'ParticleID not found: {requested_ids[~found][0]}')
            selected_rows = sorted_rows[positions]
            sorted_order = np.argsort(selected_rows)
            rows_to_read = selected_rows[sorted_order]
            restore_order = np.argsort(sorted_order)
        else:
            rows_to_read = np.empty(0, dtype=np.int64)
            restore_order = np.empty(0, dtype=np.int64)

        requested_components = list(components)
        if 'PartType0/ParticleIDs' not in requested_components:
            requested_components.append('PartType0/ParticleIDs')
        return {
            component: snapshot[component][rows_to_read][restore_order]
            if requested_ids.size else snapshot[component][:0]
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


# Physical-field loader used by the shared pipeline after SIMBA IDs are selected.
def load_physical_gas_fields(snapshot_filename: str, particle_ids, unit_factors: dict) -> dict:
    """Read selected SIMBA gas and convert it to the shared field layout.

    Returns: gas_fields (dict): aligned physical NumPy arrays keyed by field.
    """
    raw_fields = get_gas_fields_by_id(snapshot_filename, particle_ids, SIMBA_GAS_COMPONENTS)
    return convert_gas_fields(raw_fields, **unit_factors)


def get_redshift(snapshot_filename: str) -> float:
    """Read the SIMBA snapshot redshift.

    Returns: redshift (float): value stored in Header/Redshift.
    """
    with h5py.File(snapshot_filename, 'r') as snapshot:
        return float(snapshot['Header'].attrs['Redshift'])
