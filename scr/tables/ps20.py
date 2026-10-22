# A module to deal with Poeckinger & Sschaye (2020) emissivity tables. Retrieved from doi:10.1093/mnras/staa2172
import warnings

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def decode_labels(values):
    """Decode the byte strings used for line names in the PS20 HDF5 file."""
    return [
        value.decode('utf-8').strip()
        if isinstance(value, (bytes, np.bytes_))
        else str(value).strip()
        for value in np.asarray(values).ravel()
    ]


def get_ps20_metadata(filename: str) -> dict:
    """Read grid axes and line labels without loading the emissivity cube.

    The full ``Tdep/EmissivitiesVol`` dataset is multi-gigabyte. This function
    reads only its shape/type/attributes plus the small coordinate arrays.
    MetallicityBins is returned in full, including the special -50 primordial
    bin; the caller must choose its handling during interpolation.
    """
    with h5py.File(filename, 'r') as f:
        emissivity = f['Tdep/EmissivitiesVol']
        axes = {
            'redshift': f['TableBins/RedshiftBins'][...],
            'log_temperature_K': f['TableBins/TemperatureBins'][...],
            'log_metallicity_Zsun': f['TableBins/MetallicityBins'][...],
            'log_nH_cm3': f['TableBins/DensityBins'][...]}
        line_labels = decode_labels(f['IdentifierLines'][...])
        emissivity_metadata = {
            'shape': emissivity.shape,
            'dtype': str(emissivity.dtype),
            'unit': emissivity.attrs.get('Unit', ''),
            'info': emissivity.attrs.get('Info', '')}

    return {
        'axes': axes,
        'line_labels': line_labels,
        'emissivity': emissivity_metadata,
    }


def interpolate_log_emissivity(filename_table: str, log_nH, log_T, log_Z, redshift: float, line_label: str):
    """Interpolate one PS20 line for gas properties at one redshift.

    ``log_nH``, ``log_T``, and ``log_Z`` can be scalars or broadcastable
    NumPy arrays. ``line_label`` must match an entry in ``IdentifierLines``.
    The special primordial metallicity bin (the first metallicity bin, usually
    -50) is excluded. Positive gas metallicities are clipped to the regular
    metallicity grid; a warning reports counts below and above its limits.
    Temperature, density, and redshift must remain within their table ranges.

    Returns: log_emissivity (numpy.ndarray): log10(4 pi J_line) in erg cm^-3 s^-1.
    """
    with h5py.File(filename_table, 'r') as f:
        redshift_bins = f['TableBins/RedshiftBins'][...]
        temperature_bins = f['TableBins/TemperatureBins'][...]
        metallicity_bins = f['TableBins/MetallicityBins'][1:]
        density_bins = f['TableBins/DensityBins'][...]
        labels = decode_labels(f['IdentifierLines'][...])
        line_index = labels.index(line_label)
        emissivity = f['Tdep/EmissivitiesVol']

        # The snapshot has one redshift, so only the two bracketing table
        # slices are needed. The dataset axes are (redshift, T, Z, nH, line).
        if redshift < redshift_bins[0] or redshift > redshift_bins[-1]:
            raise ValueError('redshift is outside the PS20 table grid')
        upper = int(np.searchsorted(redshift_bins, redshift, side='right'))
        upper = min(max(upper, 1), len(redshift_bins) - 1)
        lower = upper - 1
        redshift_weight = (
            (redshift - redshift_bins[lower])
            / (redshift_bins[upper] - redshift_bins[lower]))                                # for Linear interpolation

        log_nH, log_T, log_Z = np.broadcast_arrays(log_nH, log_T, log_Z)
        if not np.all(np.isfinite(log_Z)):
            raise ValueError('log_Z must contain only finite values before clipping.')
        below = int(np.count_nonzero(log_Z < metallicity_bins[0]))
        above = int(np.count_nonzero(log_Z > metallicity_bins[-1]))
        if below or above:
            warnings.warn(
                f'PS20 metallicity clipping: {below} below, {above} above '
                f'the regular log10(Z/Zsun) range '
                f'[{metallicity_bins[0]:g}, {metallicity_bins[-1]:g}]. '
                'Original gas metallicities are unchanged.',
                RuntimeWarning,
                stacklevel=2,
            )
        log_Z = np.clip(log_Z, metallicity_bins[0], metallicity_bins[-1])
        points = np.column_stack((log_T.ravel(), log_Z.ravel(), log_nH.ravel()))

        lower_grid = emissivity[lower, :, 1:, :, line_index]
        upper_grid = emissivity[upper, :, 1:, :, line_index]

    lower_interpolator = RegularGridInterpolator(
        (temperature_bins, metallicity_bins, density_bins),
        lower_grid,
        bounds_error=True,
    )
    upper_interpolator = RegularGridInterpolator(
        (temperature_bins, metallicity_bins, density_bins),
        upper_grid,
        bounds_error=True,
    )
    lower_values = lower_interpolator(points)
    upper_values = upper_interpolator(points)
    values = (1 - redshift_weight) * lower_values + redshift_weight * upper_values      # Linear interpolation

    return values.reshape(log_nH.shape)


def interpolate_primordial_log_emissivity(filename_table: str, log_nH, log_T, redshift: float, line_label: str):
    """Interpolate the special PS20 zero-metallicity slice in T and nH.

    Returns: log_emissivity (numpy.ndarray): log10 volume emissivity per input point.
    """
    with h5py.File(filename_table, 'r') as f:
        redshift_bins = f['TableBins/RedshiftBins'][...]
        temperature_bins = f['TableBins/TemperatureBins'][...]
        metallicity_bins = f['TableBins/MetallicityBins'][...]
        density_bins = f['TableBins/DensityBins'][...]
        if not np.isclose(metallicity_bins[0], -50.0):
            raise ValueError('The PS20 table has no expected primordial metallicity bin.')

        labels = decode_labels(f['IdentifierLines'][...])
        line_index = labels.index(line_label)
        if redshift < redshift_bins[0] or redshift > redshift_bins[-1]:
            raise ValueError('redshift is outside the PS20 table grid')
        upper = int(np.searchsorted(redshift_bins, redshift, side='right'))
        upper = min(max(upper, 1), len(redshift_bins) - 1)
        lower = upper - 1
        redshift_weight = (
            (redshift - redshift_bins[lower])
            / (redshift_bins[upper] - redshift_bins[lower])
        )

        log_nH, log_T = np.broadcast_arrays(log_nH, log_T)
        points = np.column_stack((log_T.ravel(), log_nH.ravel()))
        emissivity = f['Tdep/EmissivitiesVol']
        lower_grid = emissivity[lower, :, 0, :, line_index]
        upper_grid = emissivity[upper, :, 0, :, line_index]

    lower_interpolator = RegularGridInterpolator(
        (temperature_bins, density_bins), lower_grid, bounds_error=True
    )
    upper_interpolator = RegularGridInterpolator(
        (temperature_bins, density_bins), upper_grid, bounds_error=True
    )
    lower_values = lower_interpolator(points)
    upper_values = upper_interpolator(points)
    values = (1 - redshift_weight) * lower_values + redshift_weight * upper_values
    return values.reshape(log_nH.shape)


def calculate_particle_line_luminosity(gas_fields: dict, *, table_filename: str,
                                       redshift: float, line_label: str) -> dict:
    """Calculate PS20 line luminosities, including its primordial metallicity bin.

    ``gas_fields`` contains physical nH, temperature, metallicity relative to
    solar, and particle volume. PS20 supplies log10 volume emissivity; multiplying
    its linear value by volume gives luminosity in erg/s.
    Exactly zero metallicity uses the primordial slice. Positive metallicity
    is clipped at the regular grid edges for interpolation only; input gas
    fields are not modified.

    Returns: emission (dict): per-particle emissivity and luminosity arrays.
    """
    required_fields = ('nH_cm3', 'T_K', 'Z_Zsun', 'volume_cm3')
    missing_fields = [name for name in required_fields if name not in gas_fields]
    if missing_fields:
        raise KeyError(f'Missing gas fields required for PS20: {missing_fields}')

    fields = {
        name: np.asarray(gas_fields[name], dtype=np.float64)
        for name in required_fields
    }
    for name, values in fields.items():
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f'{name} must be a finite one-dimensional array.')
    if len({values.size for values in fields.values()}) != 1:
        raise ValueError('PS20 input fields must have matching particle counts.')
    for name in ('nH_cm3', 'T_K', 'volume_cm3'):
        if np.any(fields[name] <= 0):
            raise ValueError(f'{name} must contain only positive values.')
    if np.any(fields['Z_Zsun'] < 0):
        raise ValueError('Z_Zsun must not contain negative values.')

    primordial = fields['Z_Zsun'] == 0
    enriched = ~primordial
    log_volume_emissivity = np.empty(primordial.size, dtype=np.float64)
    log_nH = np.log10(fields['nH_cm3'])
    log_T = np.log10(fields['T_K'])

    if np.any(primordial):
        log_volume_emissivity[primordial] = interpolate_primordial_log_emissivity(
            filename_table=table_filename,
            log_nH=log_nH[primordial],
            log_T=log_T[primordial],
            redshift=redshift,
            line_label=line_label,
        )
    if np.any(enriched):
        log_volume_emissivity[enriched] = interpolate_log_emissivity(
            filename_table=table_filename,
            log_nH=log_nH[enriched],
            log_T=log_T[enriched],
            log_Z=np.log10(fields['Z_Zsun'][enriched]),
            redshift=redshift,
            line_label=line_label,
        )

    volume_emissivity = 10.0 ** log_volume_emissivity
    return {
        'log_volume_emissivity': log_volume_emissivity,
        'volume_emissivity_erg_s_cm3': volume_emissivity,
        'line_luminosity_erg_s': volume_emissivity * fields['volume_cm3'],
    }
