# Read SIMBA gas using snapshot rows selected by catalogue membership.
from pathlib import Path

import h5py
import numpy as np

PROTON_MASS_G = 1.67262192369e-24

SIMBA_GAS_COMPONENTS = [
    'PartType0/ParticleIDs',
    'PartType0/Coordinates',
    'PartType0/Velocities',
    'PartType0/Masses',
    'PartType0/Density',
    'PartType0/InternalEnergy',
    'PartType0/Metallicity',
    'PartType0/ElectronAbundance']

def compute_conversion_factors(snapshot_filename: str, unit_inputs: dict, *,
                                to_physical: bool = True, to_units: bool = True) -> dict:
    """Keep cosmological corrections and unit-scale conversions independent.

    Raw coordinates are comoving kpc/h, masses are 1e10 Msun/h, density is
    mass per comoving code volume, and raw velocities need sqrt(a).
    to_physical selects only corrections involving a and h. to_units selects
    ordinary unit-scale conversion, retaining the comoving/h convention when
    to_physical=False. Either or both stages can be used.
    With to_units=False, mass remains in 1e10 Msun (and retains /h unless
    to_physical=True). Field names in factors are deliberately unit-neutral.
    Gamma and solar metallicity are gas-model parameters, not conversions.

    Returns: stages (dict): cosmological_factors, unit_conversions, selected
        factors by quantity, a/h metadata, and the two applied switches.
    """
    if not isinstance(to_physical, bool) or not isinstance(to_units, bool):
        raise ValueError('to_physical and to_units must be booleans.')
    required = {'solar_mass_g', 'kpc_to_cm', 'specific_energy_to_erg_g', 'z_sun_mass_fraction'}
    missing = required - unit_inputs.keys()
    unknown = unit_inputs.keys() - (required | {'gamma'})
    if missing or unknown:
        raise ValueError(f'Unit inputs: missing {sorted(missing)}; unknown {sorted(unknown)}.')
    constants = {**unit_inputs, 'gamma': unit_inputs.get('gamma', 5.0 / 3.0)}
    for name, value in constants.items():
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite.')
    if constants['gamma'] <= 1 or constants['z_sun_mass_fraction'] >= 1:
        raise ValueError('gamma must exceed one and z_sun_mass_fraction must be below one.')
    with h5py.File(snapshot_filename, 'r') as snapshot:
        a = float(snapshot['Header'].attrs['Time'])
        h = float(snapshot['Header'].attrs['HubbleParam'])
    if not np.isfinite(a) or a <= 0 or not np.isfinite(h) or h <= 0:
        raise ValueError('Snapshot scale factor a and Hubble parameter h must be positive and finite.')

    # Stage 1: dimensionless cosmological corrections; no unit-scale change.
    cosmological_factors = {
        'position': a / h,
        'mass': 1.0 / h,
        'density': h**2 / a**3,
        'velocity': float(np.sqrt(a)),
        'specific_energy': 1.0,
    }
    # Stage 2: ordinary unit conversions; no dependence on a or h.
    code_mass_g = 1.0e10 * constants['solar_mass_g']
    unit_conversions = {
        'position': 1.0,  # SIMBA's base length is already one kpc.
        'mass': code_mass_g,
        'density': code_mass_g / constants['kpc_to_cm']**3,
        'velocity': 1.0,  # SIMBA's base velocity is already one km/s.
        'specific_energy': float(constants['specific_energy_to_erg_g']),
    }
    factors = {
        quantity: (cosmological_factors[quantity] if to_physical else 1.0)
                  * (unit_conversions[quantity] if to_units else 1.0)
        for quantity in unit_conversions
    }
    return {
        'cosmological_factors': cosmological_factors,
        'unit_conversions': unit_conversions,
        'factors': factors,
        'scale_factor': a,
        'hubble_param': h,
        'to_physical': to_physical,
        'to_units': to_units,
    }


def compute_unit_factors(snapshot_filename: str, unit_inputs: dict) -> dict:
    """Select both conversion stages for the emission pipeline's physical fields.

    The combined entries keep the existing downstream interface. Both stage
    dictionaries remain available for inspection and output provenance.
    For partial conversions, use compute_conversion_factors() instead: a
    partially converted mass must not be labelled as grams by the pipeline.

    Returns: unit_factors (dict): combined physical factors, gas-model inputs,
        geometry/cosmology metadata, and both separate conversion stages.
    """
    stages = compute_conversion_factors(snapshot_filename, unit_inputs)
    factors = stages['factors']
    return {
        'position_to_physical_kpc': factors['position'],
        'mass_to_g': factors['mass'],
        'density_to_g_cm3': factors['density'],
        'velocity_to_kms': factors['velocity'],
        'specific_energy_to_erg_g': factors['specific_energy'],
        'z_sun_mass_fraction': float(unit_inputs['z_sun_mass_fraction']),
        'gamma': float(unit_inputs.get('gamma', 5.0 / 3.0)),
        'kpc_to_cm': float(unit_inputs['kpc_to_cm']),
        'scale_factor': stages['scale_factor'],
        'hubble_param': stages['hubble_param'],
        'cosmological_factors': stages['cosmological_factors'],
        'unit_conversions': stages['unit_conversions'],
    }

def select_gas_rows(catalogue_filename: str, target_index: int,
                    group_type: str = 'halo') -> np.ndarray:
    """Select snapshot gas rows from an Octavius halo.

    Returns: gas_rows (numpy.ndarray): unique integer rows in the original snapshot.
    """
    if group_type != 'halo':
        raise ValueError('group_type must be halo.')
    if (
        isinstance(target_index, (bool, np.bool_))
        or not isinstance(target_index, (int, np.integer))
        or target_index < 0
    ):
        raise ValueError('target_index must be a nonnegative integer.')
    from octavius import load_catalogue

    with load_catalogue(Path(catalogue_filename)) as catalogue:
        groups = catalogue.haloes
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

def select_cgm_gas_rows(snapshot_filename, catalogue_filename, gas_rows,
                        target_index: int, group_type: str = 'halo', *,
                        unit_factors: dict) -> np.ndarray:
    """Select halo gas within r200m and nH <= 0.13 cm^-3 using shared units.

    Coordinate/box and density conversions use the same unit_factors supplied
    to load_physical_gas_fields(). Catalogue centers and radii are already
    physical kpc. Hydrogen fraction and proton mass match convert_gas_fields().

    Returns: cgm_indices (numpy.ndarray): selected snapshot rows in input order.
    """
    if group_type != 'halo':
        raise ValueError('group_type must be halo.')
    position_factor = unit_factors['position_to_physical_kpc']
    density_factor = unit_factors['density_to_g_cm3']
    for name, factor in (('position_to_physical_kpc', position_factor),
                         ('density_to_g_cm3', density_factor)):
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f'{name} must be positive and finite.')
    from octavius import load_catalogue
    import h5py
    m_p = PROTON_MASS_G  # Same constant as convert_gas_fields(), in g
    k_B = 1.380649e-16  # erg/K

    cat = load_catalogue(catalogue_filename)
    r200m = cat.haloes.get_dataset('r200m', to_physical=True, to_units='kpc')[target_index]
    minpot_pos = cat.haloes.get_dataset('minpot_pos', to_physical=True, to_units='kpc')[target_index]
    sorted_order = np.argsort(gas_rows)
    sorted_indices = gas_rows[sorted_order]
    restore_order = np.argsort(sorted_order)

    with h5py.File(snapshot_filename, 'r') as f:
        Coor = np.asarray(
            f['PartType0/Coordinates'][sorted_indices][restore_order], dtype=np.float64
        ) * position_factor
        gas_coors = Coor[:, :3].copy()
        box_kpc = f['Header'].attrs['BoxSize'] * position_factor

        for i in range(3):
            print('The gas min coordinates:', gas_coors[:, i].min())
            gas_coors[:, i] -= minpot_pos[i]
            gas_coors[:, i] -= box_kpc * np.rint(gas_coors[:, i] / box_kpc)

        gas_radius = np.linalg.norm(gas_coors, axis=1)

        gas_Metallicities = np.asarray(
            f['PartType0/Metallicity'][sorted_indices][restore_order], dtype=np.float64
        )
        gas_Z = gas_Metallicities[:, 0] # Total metallicity
        gas_Y = gas_Metallicities[:, 1] # Mass fraction for He
        gas_X = 1.0 - gas_Z - gas_Y  # Same hydrogen fraction as convert_gas_fields()

        rho_g_cm3 = np.asarray(
            f['PartType0/Density'][sorted_indices][restore_order], dtype=np.float64
        ) * density_factor
        gas_nH = gas_X * rho_g_cm3 / m_p  # Physical hydrogen number density, cm^-3

        cgm_filter = (gas_radius <= r200m) & (gas_nH <= 0.13)
        cgm_indices = gas_rows[cgm_filter]

        return cgm_indices

def get_reference_frame(catalogue_filename: str, target_index: int,
                        group_type: str = 'halo') -> dict:
    """Read a halo's physical center and bulk velocity from Octavius.

    Halo centers use minpot_pos and velocities use com_vel_total.
    Catalogue unit conversions supply physical kpc and km/s directly.

    Returns: reference_frame (dict): center_kpc and bulk_velocity_kms 3-vectors.
    """
    if group_type != 'halo':
        raise ValueError('group_type must be halo.')
    if (
        isinstance(target_index, (bool, np.bool_))
        or not isinstance(target_index, (int, np.integer))
        or target_index < 0
    ):
        raise ValueError('target_index must be a nonnegative integer.')
    from octavius import load_catalogue

    with load_catalogue(Path(catalogue_filename)) as catalogue:
        groups = catalogue.haloes
        if groups is None or target_index >= len(groups):
            raise IndexError(f'target_index is outside the Octavius {group_type} catalogue.')
        mask = np.array([target_index], dtype=np.int64)
        center = np.asarray(groups.get_dataset(
            'minpot_pos', mask=mask, to_units='kpc', to_physical=True,
        ), dtype=np.float64)[0]  # One masked halo: unwrap (1, 3) to (3,).
        velocity = np.asarray(groups.get_dataset(
            'com_vel_total', mask=mask, to_units='km/s', to_physical=True,
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

    Returns: gas_fields (dict): aligned physical positions, velocities, mass,
        density, volume, temperature, hydrogen density, and abundance arrays.
    """
    proton_mass_g = PROTON_MASS_G
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

    Z_mass_fraction = abundances[:, 0]
    helium_mass_fraction = abundances[:, 1]
    hydrogen_mass_fraction = 1.0 - Z_mass_fraction - helium_mass_fraction
    if (not np.all(np.isfinite(abundances[:, :2]))
            or np.any(Z_mass_fraction < 0) or np.any(helium_mass_fraction < 0)
            or np.any(hydrogen_mass_fraction <= 0)):
        raise ValueError('SIMBA Z/Y mass fractions must be finite/nonnegative with X=1-Z-Y > 0.')
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
        'Z_Zsun': Z_mass_fraction / z_sun_mass_fraction,
        # Preserve the actual particle fractions for Cloudy's metal/H mapping.
        # No solar normalization, clipping or primordial substitution here.
        'Z_mass_fraction': Z_mass_fraction,
        'X_mass_fraction': hydrogen_mass_fraction,
        'C_mass_fraction': abundances[:, 2],
        'O_mass_fraction': abundances[:, 4],
    }


# Physical-field loader used by the shared pipeline after gas rows are selected.
def load_physical_gas_fields(snapshot_filename: str, gas_rows, unit_factors: dict) -> dict:
    """Read selected snapshot gas rows and convert them to physical fields.

    Returns: gas_fields (dict): aligned physical arrays, including ParticleIDs.
    """
    raw_fields = get_gas_fields(snapshot_filename, gas_rows, SIMBA_GAS_COMPONENTS)
    # Only the combined factors enter field conversion; never apply either
    # component again. Other entries describe geometry or provenance.
    gas_units = {name: value for name, value in unit_factors.items()
                 if name not in ('kpc_to_cm', 'scale_factor', 'hubble_param',
                                 'cosmological_factors', 'unit_conversions')}
    return convert_gas_fields(raw_fields, **gas_units)


def get_redshift(snapshot_filename: str) -> float:
    """Read the SIMBA snapshot redshift.

    Returns: redshift (float): value stored in Header/Redshift.
    """
    with h5py.File(snapshot_filename, 'r') as snapshot:
        return float(snapshot['Header'].attrs['Redshift'])
