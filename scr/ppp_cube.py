# This is a scirpt to generate 3D data cube (x, y, z) output

# Hey Codex, modify the function below to make a cube based on given x y z lengths, to represent the emissivity inside.

import numpy as np


def _smooth_luminosity(cube):
    """Apply a conservative, separable [1/8, 3/4, 1/8] grid kernel once per axis.

    Repeating the boundary cell implements half-sample reflection for this
    three-point stencil. No luminosity wraps to the opposite face or escapes.

    Returns: smoothed (numpy.ndarray): float64 luminosity cube in x,y,z order.
    """
    smoothed = cube
    for axis in range(3):
        padding = [(0, 0)] * 3
        padding[axis] = (1, 1)
        padded = np.pad(smoothed, padding, mode='edge')
        left, right = [slice(None)] * 3, [slice(None)] * 3
        left[axis], right[axis] = slice(0, -2), slice(2, None)
        smoothed = .125 * padded[tuple(left)] + .75 * smoothed + .125 * padded[tuple(right)]
    return smoothed


def make_ppp_cube(gas_fields: dict, particle_luminosity_erg_s, x_edges_kpc, y_edges_kpc, z_edges_kpc,
                  *, kpc_to_cm: float = 3.0856775814913673e21, smoothing: bool = True) -> dict:
    """Bin particle luminosity, then mildly smooth a physical PPP grid.

    gas_fields supplies aligned x_kpc, y_kpc, and z_kpc arrays. Coordinates
    and bin edges must already be in physical kpc and the same reference
    frame. This function does not center or periodically wrap coordinates.
    Each particle first deposits its entire luminosity in one voxel. By default
    apply a separable [1/8,3/4,1/8] kernel once along x, y and z to the LUMINOSITY
    grid, then divide by voxel volume. Its interior RMS width is 0.5 grid cells
    per axis. smoothing=False returns the original point-binned grid. Reflection
    at boundaries conserves included luminosity without wrapping or importing
    outside particles. No velocity selection or spectral broadening is applied.

    CAVEAT: this is fixed grid smoothing, not adaptive SPH reconstruction or an
    instrument PSF. On unequal bins, the physical smoothing width varies with
    grid spacing. Luminosity, not uniform emissivity, is conserved.

    Arrays use (x, y, z) order: cube[i, j, k].
    cube_erg_s stores luminosity per voxel.
    Dividing by the full physical voxel volume gives volume_emissivity_erg_s_cm3, a spatial
    volume average, not the original particle emissivity.
    Unequal bin widths are supported. The final upper edge is included, as in numpy.histogramdd.
    Outside-grid particles contribute only to the explicit exclusion totals.
    The pipeline supplies the same kpc_to_cm constant used for gas density;
    standalone calls default to the standard kpc conversion.

    Returns: cube_result (dict): luminosity/emissivity cubes, spatial edges,
        voxel volumes, smoothing provenance, axis order and particle accounting.
    """
    if not isinstance(smoothing, (bool, np.bool_)):
        raise ValueError('smoothing must be True or False.')
    positions = [np.asarray(gas_fields[f'{axis}_kpc'], dtype=np.float64)
                 for axis in ('x', 'y', 'z')]
    luminosity = np.asarray(particle_luminosity_erg_s, dtype=np.float64)
    for name, values in zip(('x_kpc', 'y_kpc', 'z_kpc', 'particle_luminosity_erg_s'),
                            [*positions, luminosity]):
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f'{name} must be a finite one-dimensional array.')
    if len({values.size for values in [*positions, luminosity]}) != 1:
        raise ValueError('Coordinates and luminosities must align by particle.')
    if np.any(luminosity < 0):
        raise ValueError('Particle luminosities must not be negative.')

    edges = []
    for axis, values in zip(('x', 'y', 'z'), (x_edges_kpc, y_edges_kpc, z_edges_kpc)):
        values = np.asarray(values, dtype=np.float64)
        if (values.ndim != 1 or values.size < 2
                or not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0)):
            raise ValueError(f'{axis}_edges_kpc must be finite, 1D, and strictly increasing.')
        edges.append(values)
    x_edges, y_edges, z_edges = edges

    # histogramdd already returns the v2 storage order (x, y, z).
    cube, _ = np.histogramdd(np.column_stack(positions), bins=edges, weights=luminosity)
    if not np.isfinite(kpc_to_cm) or kpc_to_cm <= 0:
        raise ValueError('kpc_to_cm must be positive and finite.')
    dx, dy, dz = [np.diff(values) * kpc_to_cm for values in edges]
    voxel_volume_cm3 = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
    if not np.all(np.isfinite(voxel_volume_cm3)) or np.any(voxel_volume_cm3 <= 0):
        raise ValueError('Voxel volumes must be positive and finite in cm^3.')

    inside = np.ones(luminosity.size, dtype=bool)
    for coordinate, axis_edges in zip(positions, edges):
        inside &= (coordinate >= axis_edges[0]) & (coordinate <= axis_edges[-1])

    if smoothing:
        cube = _smooth_luminosity(cube)
    included_luminosity = float(luminosity[inside].sum())
    if (not np.all(np.isfinite(cube)) or np.any(cube < 0)
            or not np.isfinite(included_luminosity)
            or not np.isclose(cube.sum(), included_luminosity, rtol=1.e-12, atol=0.)):
        raise ValueError('PPP binning/smoothing produced invalid values or failed luminosity conservation.')
    emissivity = cube / voxel_volume_cm3
    if not np.all(np.isfinite(emissivity)):
        raise ValueError('PPP volume emissivity must be finite after smoothing and volume conversion.')

    return {
        'cube_erg_s': cube,
        'volume_emissivity_erg_s_cm3': emissivity,
        'deposition': ('point binning followed by conservative separable grid smoothing; no line broadening'
                       if smoothing else 'point deposition; no smoothing or line broadening'),
        'smoothing_enabled': bool(smoothing),
        'smoothing_kernel_1d': np.array([.125, .75, .125] if smoothing else [0., 1., 0.]),
        'smoothing_sigma_voxels': .5 if smoothing else 0.,
        'smoothing_boundary': 'half-sample reflection' if smoothing else 'none',
        'smoothing_caveat': ('Fixed grid smoothing, not adaptive SPH reconstruction or an instrument PSF; '
                             'physical width varies on unequal bins; included luminosity is conserved.'
                             if smoothing else 'No spatial smoothing applied.'),
        'voxel_volume_cm3': voxel_volume_cm3,
        'x_edges_kpc': x_edges,
        'y_edges_kpc': y_edges,
        'z_edges_kpc': z_edges,
        'cube_axis_order': 'x,y,z',
        'input_particle_count': luminosity.size,
        'included_particle_count': int(np.count_nonzero(inside)),
        'excluded_particle_count': int(np.count_nonzero(~inside)),
        'input_luminosity_erg_s': float(luminosity.sum()),
        'included_luminosity_erg_s': included_luminosity,
        'excluded_luminosity_erg_s': float(luminosity[~inside].sum()),
    }
