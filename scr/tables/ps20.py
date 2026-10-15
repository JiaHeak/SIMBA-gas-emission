# A module to deal with Poeckinger & Sschaye (2020) emissivity tables. Retrieved from doi:10.1093/mnras/staa2172
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
    -50) is excluded; this review version interpolates only the regular
    metallicity grid.

    Returns the table's log10 volume emissivity values. For the PS20 table
    discussed in this project, these are log10(4 pi J_line) in
    erg cm^-3 s^-1.
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
