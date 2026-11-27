"""Map processing and plotting, independent of catalogue loading/selection."""

import warnings

import numpy as np


def _smoothing_kernel(smoothing, sigma_pixels):
    """Resolve compact, sampled Gaussian, or disabled grid smoothing.

    sigma_pixels sets the Gaussian width only; compact smoothing retains its
    fixed RMS 0.5 pixel. The Gaussian radius is ceil(4*sigma), at least one pixel.

    Returns: configuration (tuple[str, numpy.ndarray, float]): method, normalized
        odd-length 1D kernel, and nominal sigma in pixels (zero when disabled).
    """
    if isinstance(smoothing, (bool, np.bool_)):
        method = 'compact' if smoothing else 'none'
    elif isinstance(smoothing, str) and smoothing in ('compact', 'gaussian'):
        method = smoothing
    else:
        raise ValueError("smoothing must be True, False, 'compact', or 'gaussian'.")
    if (isinstance(sigma_pixels, (bool, np.bool_))
            or not isinstance(sigma_pixels, (int, float, np.integer, np.floating))
            or not np.isfinite(sigma_pixels) or sigma_pixels <= 0):
        raise ValueError('sigma_pixels must be a positive finite number.')
    sigma = float(sigma_pixels)
    if method != 'gaussian' and sigma != .5:
        raise ValueError("sigma_pixels is configurable only with smoothing='gaussian'.")
    if method == 'none':
        return method, np.array([1.]), 0.
    if method == 'compact':
        return method, np.array([.125, .75, .125]), .5
    if sigma > np.finfo(float).max / 4:
        raise ValueError('sigma_pixels is too large to construct a finite Gaussian radius.')
    radius = max(1, int(np.ceil(4*sigma)))
    offsets = np.arange(-radius, radius+1, dtype=float)
    with np.errstate(over='ignore', under='ignore'):
        kernel = np.exp(-.5*(offsets/sigma)**2)
    kernel /= kernel.sum()
    return method, kernel, sigma


def _smooth_grid(values, kernel):
    """Apply a separable kernel with half-sample reflection at the image edges.

    Returns: values (numpy.ndarray): smoothed grid.
    """
    radius = len(kernel) // 2
    if radius:
        for axis in range(2):
            padding = [(0, 0), (0, 0)]
            padding[axis] = (radius, radius)
            padded = np.pad(values, padding, mode='symmetric')
            filtered = np.zeros_like(values)
            for offset, weight in enumerate(kernel):
                window = [slice(None)] * 2
                window[axis] = slice(offset, offset+values.shape[axis])
                filtered += weight*padded[tuple(window)]
            values = filtered
    return values


def _smoothing_metadata(method, kernel, sigma):
    """Describe the selected grid smoothing kernel.

    Returns: metadata (dict): kernel, width and boundary settings.
    """
    radius = len(kernel) // 2
    offsets = np.arange(-radius, radius+1, dtype=float)
    return dict(smoothing_enabled=method != 'none',
                     smoothing_method=method, smoothing_kernel_1d=kernel,
                     smoothing_sigma_pixels=sigma,
                     smoothing_effective_sigma_pixels=float(np.sqrt(np.sum(kernel*offsets**2))),
                     smoothing_radius_pixels=radius,
                     smoothing_truncate_sigma=4. if method == 'gaussian' else None,
                     smoothing_boundary='half-sample reflection' if method != 'none' else 'none',
                     smoothing_caveat=('Fixed grid smoothing, not adaptive SPH or an instrument PSF; '
                                       'physical width varies on unequal bins.' if method != 'none' else 'No smoothing.'))


def smooth_map(result, enabled=True, *, sigma_pixels=.5):
    """Smooth binned luminosity with a normalized reflecting kernel.

    Returns: processed (dict): copied map with values, smoothing and coverage metadata.
    """
    method, kernel, sigma = _smoothing_kernel(enabled, sigma_pixels)
    values = np.array(result['values'], dtype=np.float64, copy=True)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Map luminosities must be finite, nonnegative and two-dimensional.')
    values = _smooth_grid(values, kernel)
    if (not np.all(np.isfinite(values))
            or not np.isclose(values.sum(), result['included_luminosity_erg_s'], rtol=1e-10, atol=0)):
        raise ValueError('Map smoothing failed included-luminosity conservation.')
    return dict(result, values=values, **_smoothing_metadata(method, kernel, sigma))


def finish_weighted_map(result, configuration):
    """Smooth both additive grids before dividing them into a weighted mean.

    Returns: image (dict): weighted values, additive grids, mask and smoothing metadata.
    """
    method, kernel, sigma = configuration
    weighted_sum = _smooth_grid(result['weighted_sum'], kernel)
    weight_sum = _smooth_grid(result['weight_sum'], kernel)
    if (not np.all(np.isfinite(weighted_sum)) or not np.all(np.isfinite(weight_sum))
            or not np.isclose(weight_sum.sum(), result['included_weight'], rtol=1e-10, atol=0)):
        raise ValueError('Weighted map failed finite sums or weight conservation.')
    valid = weight_sum > 0
    values = np.divide(weighted_sum, weight_sum, out=np.full(weight_sum.shape, np.nan), where=valid)
    if not np.all(np.isfinite(values[valid])):
        raise ValueError('Nonfinite weighted mean.')
    return dict(result, values=values, valid_mask=valid, weighted_sum=weighted_sum,
                weight_sum=weight_sum, **_smoothing_metadata(method, kernel, sigma))


def plot_map_result(result, *, ax=None, log=None, cmap='magma'):
    """Draw a prepared luminosity, surface-brightness, line-ratio or weighted map.

    Undefined line ratios (NaN) are masked. Zeros are masked for log display.
    Maps without positive values warn and use linear colors in log mode.

    Returns: ax (matplotlib.axes.Axes): plotted axes; caller controls display/save.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    values = np.asarray(result['values'])
    edges = result['edges']
    is_ratio = result.get('quantity') == 'line_ratio'
    is_surface_brightness = result.get('field') == 'surface_brightness'
    is_weighted = 'weighted_by' in result
    is_velocity = is_weighted and result['quantity'] == 'velocity'
    if log is None:
        log = not is_velocity
    if (values.ndim != 2 or len(edges) != 2
            or values.shape != tuple(len(edge)-1 for edge in edges)
            or np.any(np.isinf(values)) or (not (is_ratio or is_weighted) and np.any(np.isnan(values)))
            or (not is_velocity and np.any(values < 0))):
        raise ValueError('Invalid map values or bin-edge dimensions.')
    if log and is_velocity and np.any(values < 0):
        raise ValueError('Use log=False to display signed velocities.')
    positive = values[values > 0]
    norm = None
    display = np.ma.masked_invalid(values)
    if log and positive.size:
        low, high = float(positive.min()), float(positive.max())
        if low == high:
            # A constant positive map can still be shown on a logarithmic scale.
            low, high = low / 10., high * 10.
        norm = LogNorm(vmin=low, vmax=high)
        display = np.ma.masked_equal(display, 0)
    elif log:
        message = 'Map has no positive values' if is_ratio or is_weighted else 'All-zero map'
        warnings.warn(f'{message}; using linear colors.', RuntimeWarning, stacklevel=2)
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    mesh = ax.pcolormesh(*edges, display.T, shading='auto', norm=norm, cmap=cmap)
    colorbar_label = 'Line ratio [dimensionless]' if is_ratio else 'Luminosity per pixel [erg/s]'
    if is_surface_brightness:
        colorbar_label = f"Surface brightness [{result['unit']}]"
    if is_weighted:
        quantity = 'Temperature' if result['quantity'] == 'temperature' else 'LOS velocity'
        colorbar_label = f"{result['weighted_by'].capitalize()}-weighted {quantity.lower()} [{result['unit']}]"
    ax.figure.colorbar(mesh, ax=ax, label=colorbar_label)
    ax.set(xlabel=result['axis_names'][0]+' offset [kpc]',
           ylabel=result['axis_names'][1]+' offset [kpc]')
    ax.set_aspect('equal')
    title = result['line']
    if is_surface_brightness:
        title += f"\nz = {result['redshift']:g}"
    if is_weighted:
        title = colorbar_label
        if result['weighted_by'] == 'luminosity':
            title += f"\n{result['line']}"
    if result.get('smoothing_enabled'):
        if result.get('smoothing_method') == 'gaussian':
            title += f'\nGaussian smoothing: sigma {result["smoothing_sigma_pixels"]:g} pixel per axis'
        else:
            title += '\nGrid smoothing: RMS 0.5 pixel per axis'
    ax.set_title(title)
    return ax
