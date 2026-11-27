"""Lazy particle catalogue access and chunked luminosity maps/velocity spectra."""
from copy import deepcopy
from pathlib import Path
import json
import re
import h5py
import numpy as np
from .map_tools import _smoothing_kernel, finish_weighted_map, smooth_map, plot_map_result
from .observables import SURFACE_BRIGHTNESS_UNIT, surface_brightness_factor
from .raw_snapshot import RawSnapshot

SCHEMA = 'codex_simba_particle_emission_v1'
PARTICLE_UNITS = dict(snapshot_row='dimensionless', particle_id='dimensionless', position_kpc='kpc',
                      velocity_kms='km s^-1', volume_cm3='cm^3', smoothing_length_kpc='kpc')
LINE_UNITS = {'luminosity_erg_s': 'erg s^-1', 'volume_emissivity_erg_s_cm3': 'erg s^-1 cm^-3'}
SNAPSHOT_FIELDS = {'temperature': ('T_K', 'K'), 'density': ('rho_g_cm3', 'g cm^-3')}
LIGHT_SPEED_KMS = 299792.458
PROFILE_UNITS = {'luminosity': 'erg s^-1', 'cumulative_luminosity': 'erg s^-1',
                 'emission': 'erg s^-1 cm^-3'}


def _edges(values):
    """Validate explicit bin edges.

    Returns: edges (numpy.ndarray): finite, increasing one-dimensional edges.
    """
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError('Edges must be finite and strictly increasing.')
    return edges


def _vector(value):
    """Validate a physical centre or bulk velocity.

    Returns: vector (numpy.ndarray): finite three-component vector.
    """
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError('Reference must contain three finite components.')
    return vector


def _map_edges(half_width_kpc, bins, x_edges_kpc, y_edges_kpc):
    """Construct the two projected axes from explicit edges or a symmetric width.

    Returns: edges (list[numpy.ndarray]): increasing physical pixel edges.
    """
    if not np.isfinite(half_width_kpc) or half_width_kpc <= 0:
        raise ValueError('half_width_kpc must be positive.')
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError('bins must be a positive integer.')
    default = np.linspace(-half_width_kpc, half_width_kpc, bins + 1)
    return [default if edge is None else _edges(edge) for edge in (x_edges_kpc, y_edges_kpc)]


def _divide_line_values(numerator, denominator, shape, min_denominator):
    """Divide aligned luminosities and mark bins with undefined ratios.

    Returns: division (tuple[numpy.ndarray, numpy.ndarray]): ratios and validity mask.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.shape != shape or denominator.shape != shape:
        raise ValueError('Each line must have one value per bin.')
    for values in (numerator, denominator):
        if np.any(np.isinf(values)) or np.any(values < 0):
            raise ValueError('Line values must be nonnegative and finite or NaN for masked bins.')
    if min_denominator is None:
        min_denominator = 0.
    if (isinstance(min_denominator, (bool, np.bool_))
            or not isinstance(min_denominator, (int, float, np.integer, np.floating))
            or not np.isfinite(min_denominator) or min_denominator < 0):
        raise ValueError('min_denominator must be finite and nonnegative.')
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    valid &= (denominator > 0) & (denominator >= min_denominator)
    with np.errstate(over='ignore'):
        ratio = np.divide(numerator, denominator, out=np.full(shape, np.nan), where=valid)
    if not np.all(np.isfinite(ratio[valid])):
        raise ValueError('Nonfinite line ratio (possible numerical overflow).')
    return ratio, valid


def _print_product_units(product):
    """Print units for a prepared product's values, axes and optional weights.

    Returns: result (NoneType): prints a single line of unit information.
    """
    units = [f"{product['quantity']}: {product['unit']}"]
    units.extend(f'{name}: {unit}' for name, unit in
                 zip(product['axis_names'], product['axis_units']))
    if 'weight_unit' in product:
        units.append(f"{product['weighted_by']} weights: {product['weight_unit']}")
    print(' | '.join(units))


class ParticleEmissionCatalogue:
    """A read-only catalogue; constructor reads metadata, not particle arrays.

    All public calls accept verbose=True to print the units of their fields
    or products. The default is quiet.
    """

    def __init__(self, filename, *, verbose=False):
        """Validate layout and load small metadata.

        Set verbose=True to print the saved particle, line and reference units.

        Returns: result (NoneType): initializes the catalogue interface.
        """
        self.filename = Path(filename).resolve()
        with h5py.File(self.filename, 'r') as source:
            if source.attrs.get('schema') != SCHEMA or source.attrs.get('complete') != True:
                raise ValueError('Expected a complete particle emission catalogue.')
            self.metadata = dict(source.attrs)
            self.particle_count = int(source.attrs['particle_count'])
            self.line_labels = tuple(source['lines/labels'].asstr()[...])
            if not self.line_labels or len(set(self.line_labels)) != len(self.line_labels):
                raise ValueError('Invalid line labels.')
            if len(self.line_labels) != source.attrs['line_count']:
                raise ValueError('Line count mismatch.')
            field_units = {}
            for name, unit in PARTICLE_UNITS.items():
                shape = (self.particle_count, 3) if name in ('position_kpc', 'velocity_kms') else (self.particle_count,)
                dataset = source['particles/' + name]
                if dataset.shape != shape or dataset.attrs.get('unit') != unit:
                    raise ValueError(f'Invalid particle layout/units: {name}')
                if name in ('particle_id', 'snapshot_row') and not np.issubdtype(dataset.dtype, np.integer):
                    raise ValueError(f'{name} must retain integer precision.')
                field_units[name] = dataset.attrs['unit']
            reference = source['reference_frame']
            self._reference_units = {}
            for name in ('center_kpc', 'bulk_velocity_kms', 'halo_center_kpc', 'halo_bulk_velocity_kms', 'box_size_kpc'):
                unit = 'km s^-1' if 'velocity' in name else 'kpc'
                if reference[name].attrs.get('unit') != unit:
                    raise ValueError(f'Invalid reference units: {name}')
                self._reference_units[name] = reference[name].attrs['unit']
            self.center_kpc = _vector(reference['center_kpc'][...])
            self.bulk_velocity_kms = _vector(reference['bulk_velocity_kms'][...])
            self.halo_center_kpc = _vector(reference['halo_center_kpc'][...])
            self.halo_bulk_velocity_kms = _vector(reference['halo_bulk_velocity_kms'][...])
            self.box_size_kpc = float(reference['box_size_kpc'][()])
            self.periodic = bool(reference.attrs['periodic'])
            if not np.isfinite(self.box_size_kpc) or self.box_size_kpc <= 0:
                raise ValueError('Invalid box size.')
            self.diagnostics = {}
            for index, label in enumerate(self.line_labels):
                group = source[f'lines/line_{index:04d}']
                if group.attrs['line_label'] != label:
                    raise ValueError('Line label/group mismatch.')
                for name, unit in LINE_UNITS.items():
                    if group[name].shape != (self.particle_count,) or group[name].attrs.get('unit') != unit:
                        raise ValueError('Invalid line field layout/units.')
                    if index == 0:
                        field_units[name] = group[name].attrs['unit']
                self.diagnostics[label] = json.loads(group.attrs['diagnostics_json'])
            self.provenance = dict(source['provenance'].attrs)
            self.table_metadata = json.loads(source['provenance/table_metadata_json'].asstr()[()])
            self.parameters_yaml = source['provenance/parameters_yaml'].asstr()[()]
        if verbose:
            for name, unit in {**field_units, **self._reference_units}.items():
                print(f'{name}: {unit}')

    def describe(self, *, verbose=False):
        """Summarize available particles, lines and reference settings.

        Set verbose=True to print the units of the physical frame fields.

        Returns: summary (dict): particle count, line labels and physical frame.
        """
        if verbose:
            for name in ('center_kpc', 'bulk_velocity_kms', 'box_size_kpc'):
                print(f'{name}: {self._reference_units[name]}')
        return dict(particle_count=self.particle_count, line_labels=self.line_labels,
                    center_kpc=self.center_kpc.copy(), bulk_velocity_kms=self.bulk_velocity_kms.copy(),
                    box_size_kpc=self.box_size_kpc, periodic=self.periodic)

    def get_fields(self, field, *, line=None, selection=slice(None), snapshot=None, verbose=False):
        """Read one field for the requested catalogue rows.

        For 'temperature' (K) or 'density' (physical mass density in g/cm³),
        pass the original snapshot via snapshot=read_RawSnapshot(...). Both
        use the catalogue's saved conversion factors and a catalogue row
        selection. Set verbose=True to print the unit.

        Returns: values (numpy.ndarray or numpy scalar): selected particle values.
        """
        if field in SNAPSHOT_FIELDS:
            raw_name, unit = SNAPSHOT_FIELDS[field]
            values = self._get_snapshot_fields(snapshot, selection, (raw_name,))[raw_name]
            if verbose:
                print(f'{field}: {unit}')
            return values

        if field in PARTICLE_UNITS:
            path = 'particles/' + field
        elif field in LINE_UNITS:
            if line not in self.line_labels:
                raise ValueError(f'Choose an exact line label from {self.line_labels}.')
            path = f'lines/line_{self.line_labels.index(line):04d}/' + field
        else:
            raise KeyError(field)
        with h5py.File(self.filename, 'r') as source:
            dataset = source[path]
            values = dataset[selection]
            if verbose:
                print(f"{field}: {dataset.attrs['unit']}")
            return values

    def _get_snapshot_fields(self, snapshot, selection, fields):
        """Resolve a catalogue selection and read its snapshot fields together.

        Returns: values (dict): independent selected arrays or scalars.
        """
        if not isinstance(snapshot, RawSnapshot):
            raise TypeError('Snapshot fields require snapshot=read_RawSnapshot(...).')
        with h5py.File(self.filename, 'r') as source:
            factors = self._snapshot_unit_factors(source, snapshot)
            rows = source['particles/snapshot_row'][selection]
            ids = source['particles/particle_id'][selection]
        values = self._read_snapshot_fields(
            snapshot, np.atleast_1d(rows), np.atleast_1d(ids), fields, factors)
        if np.ndim(rows) == 0:
            return {name: field_values[0] for name, field_values in values.items()}
        return values

    @staticmethod
    def _snapshot_unit_factors(source, snapshot):
        """Check the snapshot epoch against the saved catalogue conversion factors.

        Returns: factors (dict): catalogue conversion factors for the raw snapshot.
        """
        factors = json.loads(source['provenance/unit_factors_json'].asstr()[()])
        if (not np.isclose(snapshot.scale_factor, factors['scale_factor'], rtol=1e-10, atol=0)
                or not np.isclose(snapshot.hubble_param, factors['hubble_param'], rtol=1e-10, atol=0)):
            raise ValueError('Snapshot scale factor or Hubble parameter differs from the catalogue.')
        return factors

    @staticmethod
    def _read_snapshot_fields(snapshot, rows, ids, fields, factors):
        """Read aligned gas fields in bounded HDF5 batches and check particle IDs.

        Returns: values (dict[str, numpy.ndarray]): fields in catalogue selection order.
        """
        values = {field: np.empty(len(rows)) for field in fields}
        for start in range(0, len(rows), 4096):
            batch = slice(start, start + 4096)
            raw = snapshot.read_gas_fields(rows[batch], fields=fields,
                                           unit_factors=factors, particle_ids=ids[batch])
            for field in fields:
                values[field][batch] = raw[field]
        return values

    def _spatial_reference(self, center, periodic, chunk_size):
        """Resolve and validate the spatial settings shared by catalogue renders.

        Returns: reference (tuple): physical center (numpy.ndarray), periodic flag (bool).
        """
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer.')
        if isinstance(center, str) and center == 'halo':
            center = self.halo_center_kpc
        center = self.center_kpc if center is None else _vector(center)
        periodic = self.periodic if periodic is None else periodic
        if not isinstance(periodic, (bool, np.bool_)):
            raise ValueError('periodic must be boolean.')
        return center, bool(periodic)

    def _position_offsets(self, positions, selection, center, periodic):
        """Read centered positions for one chunk with optional minimum-image wrapping.

        Returns: offsets (numpy.ndarray): particle coordinates relative to center in kpc.
        """
        offsets = positions[selection] - center
        if not np.all(np.isfinite(offsets)):
            raise ValueError('Invalid particle coordinates.')
        if periodic:
            offsets = (offsets + self.box_size_kpc/2) % self.box_size_kpc - self.box_size_kpc/2
        return offsets

    def _render(self, line, edges, los_axis, center, bulk_velocity, periodic, chunk_size, aperture_radius_kpc,
                spectrum: bool, *, rest_wavelength_angstrom=None):
        """Accumulate luminosity from aligned chunks in a chosen frame.

        Returns: result (dict): bins, frame, units and conservation accounting.
        """
        if line not in self.line_labels or los_axis not in ('x', 'y', 'z'):
            raise ValueError('Choose an available line and LOS x, y or z.')
        center, periodic = self._spatial_reference(center, periodic, chunk_size)
        if isinstance(bulk_velocity, str) and bulk_velocity == 'halo':
            bulk_velocity = self.halo_bulk_velocity_kms
        bulk_velocity = self.bulk_velocity_kms if bulk_velocity is None else _vector(bulk_velocity)
        if aperture_radius_kpc is not None and (not np.isfinite(aperture_radius_kpc) or aperture_radius_kpc <= 0):
            raise ValueError('Aperture radius must be positive and finite.')
        edges = [_edges(e) for e in edges]
        axis = 'xyz'.index(los_axis)
        image_axes = [i for i in range(3) if i != axis]
        values = np.zeros(tuple(len(e)-1 for e in edges))
        included_luminosity = excluded_luminosity = 0.
        included_count = 0
        with h5py.File(self.filename, 'r') as source:
            luminosities = source[f'lines/line_{self.line_labels.index(line):04d}/luminosity_erg_s']
            positions = source['particles/position_kpc']
            for start in range(0, self.particle_count, chunk_size):
                selection = slice(start, start + chunk_size)
                position = self._position_offsets(positions, selection, center, periodic)
                luminosity = luminosities[selection]
                if not np.all(np.isfinite(luminosity)) or np.any(luminosity < 0):
                    raise ValueError('Invalid particle luminosities.')
                coordinates = position[:, image_axes]
                keep = np.ones(len(position), dtype=bool)
                if aperture_radius_kpc is not None:
                    keep &= np.linalg.norm(coordinates, axis=1) <= aperture_radius_kpc
                if spectrum:
                    coordinates = source['particles/velocity_kms'][selection, axis:axis+1] - bulk_velocity[axis]
                    if not np.all(np.isfinite(coordinates)):
                        raise ValueError('Invalid particle velocities.')
                    if rest_wavelength_angstrom is not None:
                        if np.any(np.abs(coordinates) >= LIGHT_SPEED_KMS):
                            raise ValueError('Wavelength rendering requires subluminal LOS velocities.')
                        coordinates = rest_wavelength_angstrom * (1 + coordinates / LIGHT_SPEED_KMS)
                        if not np.all(np.isfinite(coordinates)):
                            raise ValueError('Nonfinite Doppler-shifted wavelengths.')
                for index, edge in enumerate(edges):
                    keep &= (coordinates[:, index] >= edge[0]) & (coordinates[:, index] <= edge[-1])
                values += np.histogramdd(coordinates[keep], bins=edges, weights=luminosity[keep])[0]
                included_luminosity += luminosity[keep].sum()
                excluded_luminosity += luminosity[~keep].sum()
                included_count += np.count_nonzero(keep)
        if (not np.all(np.isfinite(values)) or not np.all(np.isfinite([included_luminosity, excluded_luminosity]))
                or not np.isclose(values.sum(), included_luminosity, rtol=1e-10, atol=0)):
            raise ValueError('Rendering failed luminosity conservation.')
        return dict(values=values, edges=edges, unit='erg s^-1',
                    product_type='spectrum' if spectrum else 'map',
                    quantity='luminosity per velocity bin' if spectrum else 'luminosity per pixel',
                    axis_units=['km s^-1'] if spectrum else ['kpc', 'kpc'],
                    axis_names=['velocity'] if spectrum else ['xyz'[i] for i in image_axes],
                    line=line, los_axis=los_axis, center_kpc=np.array(center), bulk_velocity_kms=np.array(bulk_velocity),
                    periodic=bool(periodic), aperture_radius_kpc=aperture_radius_kpc,
                    included_luminosity_erg_s=float(included_luminosity), excluded_luminosity_erg_s=float(excluded_luminosity),
                    included_particle_count=int(included_count), excluded_particle_count=self.particle_count-int(included_count))

    def make_map(self, line, *, field='luminosity', redshift=None, los_axis='z', half_width_kpc=100., bins=128,
                 x_edges_kpc=None, y_edges_kpc=None, center=None, periodic=None,
                 chunk_size=100000, smoothing=False, sigma_pixels=.5, verbose=False):
        """Project line luminosity or observed surface brightness along the full LOS.

        Use True or 'compact' for 0.5-pixel RMS grid smoothing, or 'gaussian'
        for a sampled Gaussian with a width of sigma_pixels. The default is
        point binning. Explicit edges refer to the two image axes selected by
        los_axis. Surface brightness uses each physical pixel area and the
        catalogue redshift unless explicitly overridden.

        Returns: result (dict): values (numpy.ndarray, image-axis-first), edges,
            units, reference settings, smoothing metadata and inclusion accounting.
        """
        if field not in ('luminosity', 'surface_brightness'):
            raise ValueError("field must be 'luminosity' or 'surface_brightness'.")
        if field == 'luminosity' and redshift is not None:
            raise ValueError("redshift applies only to field='surface_brightness'.")
        _smoothing_kernel(smoothing, sigma_pixels)  # Validate before reading particles.
        edges = _map_edges(half_width_kpc, bins, x_edges_kpc, y_edges_kpc)
        if field == 'surface_brightness':
            redshift = self.metadata.get('redshift') if redshift is None else redshift
            factor = surface_brightness_factor(redshift)
            pixel_area_kpc2 = np.multiply.outer(np.diff(edges[0]), np.diff(edges[1]))
            if not np.all(np.isfinite(pixel_area_kpc2) & (pixel_area_kpc2 > 0)):
                raise ValueError('Physical pixel areas must be finite and positive.')
        result = self._render(line, edges,
                              los_axis, center, None, periodic, chunk_size, None, False)
        result = smooth_map(result, enabled=smoothing, sigma_pixels=sigma_pixels)
        result['field'] = field
        if field == 'surface_brightness':
            # Smooth conserved luminosity first, then divide by each pixel's physical area.
            with np.errstate(over='ignore', invalid='ignore'):
                values = result['values'] / pixel_area_kpc2 * factor
            if not np.all(np.isfinite(values)):
                raise ValueError('Nonfinite surface brightness after conversion.')
            result.update(values=values, quantity='surface brightness',
                          unit=SURFACE_BRIGHTNESS_UNIT, redshift=float(redshift),
                          pixel_area_kpc2=pixel_area_kpc2)
        if verbose:
            _print_product_units(result)
        return result

    def make_weighted_map(self, line=None, los_axis='z', half_width_kpc=100., *,
                          quantity='temperature', weighted_by='mass', bins=128,
                          x_edges_kpc=None, y_edges_kpc=None, center=None, velocity_frame=None,
                          periodic=None, chunk_size=100000, smoothing=False, sigma_pixels=.5,
                          snapshot=None, verbose=False):
        """Project a weighted mean temperature or signed LOS velocity.

        Choose mass or a saved line's luminosity as weights. Velocity uses the
        saved bulk velocity by default, or 'halo' or an explicit vector.
        Supply a RawSnapshot for temperature or mass weighting. Its fields
        are read only for particles inside the image, using saved unit factors.

        input snapshot should be a RawSnapshot object.

        Returns: image (dict): editable weighted means, additive grids, and mask.
        """
        if quantity not in ('temperature', 'velocity'):
            raise ValueError("quantity must be 'temperature' or 'velocity'.")
        if weighted_by not in ('mass', 'luminosity'):
            raise ValueError("weighted_by must be 'mass' or 'luminosity'.")
        if los_axis not in ('x', 'y', 'z'):
            raise ValueError('Choose LOS x, y or z.')
        if weighted_by == 'luminosity' and line is None:
            raise ValueError('Specify a line label for luminosity weighting.')
        if line is not None and line not in self.line_labels:
            raise ValueError(f'Choose an exact line label from {self.line_labels}.')
        configuration = _smoothing_kernel(smoothing, sigma_pixels)
        edges = _map_edges(half_width_kpc, bins, x_edges_kpc, y_edges_kpc)
        center, periodic = self._spatial_reference(center, periodic, chunk_size)
        if isinstance(velocity_frame, str) and velocity_frame == 'halo':
            velocity_frame = self.halo_bulk_velocity_kms
        bulk_velocity = self.bulk_velocity_kms if velocity_frame is None else _vector(velocity_frame)
        axis = 'xyz'.index(los_axis)
        image_axes = [i for i in range(3) if i != axis]
        shape = tuple(len(edge) - 1 for edge in edges)
        weighted_sum = np.zeros(shape)
        weight_sum = np.zeros(shape)
        included_count = 0
        included_weight = 0.
        fields = set()
        if quantity == 'temperature':
            fields.add('T_K')
        if weighted_by == 'mass':
            fields.add('mass_g')
        if fields and not isinstance(snapshot, RawSnapshot):
            raise TypeError('Temperature and mass weighting require snapshot=read_RawSnapshot(...).')

        with h5py.File(self.filename, 'r') as source:
            particles = source['particles']
            if fields:
                factors = self._snapshot_unit_factors(source, snapshot)
            if weighted_by == 'luminosity':
                luminosities = source[f'lines/line_{self.line_labels.index(line):04d}/luminosity_erg_s']
            for start in range(0, self.particle_count, chunk_size):
                selection = slice(start, start + chunk_size)
                offsets = self._position_offsets(particles['position_kpc'], selection, center, periodic)
                coordinates = offsets[:, image_axes]
                inside = np.ones(len(coordinates), dtype=bool)
                for index, edge in enumerate(edges):
                    inside &= (coordinates[:, index] >= edge[0]) & (coordinates[:, index] <= edge[-1])
                if not np.any(inside):
                    continue
                if fields:
                    rows = particles['snapshot_row'][selection][inside]
                    ids = particles['particle_id'][selection][inside]
                    values = self._read_snapshot_fields(snapshot, rows, ids, fields, factors)
                if quantity == 'temperature':
                    samples = values['T_K']
                else:
                    samples = particles['velocity_kms'][selection, axis][inside] - bulk_velocity[axis]
                weights = values['mass_g'] if weighted_by == 'mass' else luminosities[selection][inside]
                if not np.all(np.isfinite(samples)) or (quantity == 'temperature' and np.any(samples <= 0)):
                    raise ValueError(f'Invalid particle {quantity} in weighted map.')
                if (not np.all(np.isfinite(weights)) or np.any(weights < 0)
                        or (weighted_by == 'mass' and np.any(weights == 0))):
                    raise ValueError(f'Invalid particle {weighted_by} weights.')
                coordinates = coordinates[inside]
                weighted_sum += np.histogramdd(coordinates, bins=edges, weights=weights * samples)[0]
                weight_sum += np.histogramdd(coordinates, bins=edges, weights=weights)[0]
                included_count += len(samples)
                included_weight += float(weights.sum())

        result = dict(product_type='map', quantity=quantity, weighted_by=weighted_by,
                      unit='K' if quantity == 'temperature' else 'km s^-1',
                      weight_unit='g' if weighted_by == 'mass' else 'erg s^-1',
                      line=line if weighted_by == 'luminosity' else None,
                      edges=edges, axis_names=['xyz'[i] for i in image_axes], axis_units=['kpc', 'kpc'],
                      los_axis=los_axis, center_kpc=center.copy(), periodic=periodic,
                      bulk_velocity_kms=bulk_velocity.copy(), weighted_sum=weighted_sum, weight_sum=weight_sum,
                      included_weight=included_weight, included_particle_count=included_count,
                      excluded_particle_count=self.particle_count - included_count)
        if fields:
            result['snapshot_filename'] = str(snapshot.filename)
        result = finish_weighted_map(result, configuration)
        if verbose:
            _print_product_units(result)
        return result

    def make_phase_diagram(self, *, selection=slice(None), snapshot=None, bins=80, verbose=False):
        """Bin selected particles by physical gas density and temperature.

        Selection uses catalogue rows, including a Boolean mask of length
        particle_count. True includes a particle. The saved snapshot path is
        used by default. Pass a RawSnapshot if that file has moved. Each bin
        counts particles, with equally spaced logarithmic edges on both axes.

        Returns: phase (dict): counts indexed by density then temperature,
            with physical bin edges and units.
        """
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError('bins must be a positive integer.')
        if snapshot is None:
            snapshot = RawSnapshot(self.provenance['snapshot_filename'])
        fields = self._get_snapshot_fields(snapshot, selection, ('rho_g_cm3', 'T_K'))
        density = np.atleast_1d(fields['rho_g_cm3'])
        temperature = np.atleast_1d(fields['T_K'])
        if not density.size:
            raise ValueError('The phase diagram selection contains no particles.')
        if np.any(density <= 0) or np.any(temperature <= 0):
            raise ValueError('A logarithmic phase diagram requires positive density and temperature.')

        counts, log_density_edges, log_temperature_edges = np.histogram2d(
            np.log10(density), np.log10(temperature), bins=bins)
        result = dict(product_type='phase_diagram', quantity='particle count per bin',
                      values=counts.astype(np.int64),
                      edges=[10**log_density_edges, 10**log_temperature_edges],
                      axis_names=['density', 'temperature'],
                      axis_units=[SNAPSHOT_FIELDS[name][1] for name in ('density', 'temperature')],
                      unit='particles', particle_count=len(density))
        if verbose:
            _print_product_units(result)
        return result

    def plot_phase_diagram(self, phase, *, ax=None, cmap='viridis', show=True, verbose=False):
        """Plot a prepared density–temperature diagram with logarithmic axes.

        Returns: ax (matplotlib.axes.Axes): plotted axes for customization or saving.
        """
        import matplotlib.pyplot as plt

        if not isinstance(phase, dict) or phase.get('product_type') != 'phase_diagram':
            raise ValueError('Expected a phase diagram from make_phase_diagram().')
        density_edges, temperature_edges = (_edges(edges) for edges in phase['edges'])
        counts = np.asarray(phase['values'])
        if (density_edges[0] <= 0 or temperature_edges[0] <= 0
                or counts.shape != (len(density_edges) - 1, len(temperature_edges) - 1)
                or not np.all(np.isfinite(counts)) or np.any(counts < 0)):
            raise ValueError('Expected positive bin edges and finite nonnegative counts on the phase grid.')
        if ax is None:
            _, ax = plt.subplots()
        image = ax.pcolormesh(density_edges, temperature_edges, counts.T, cmap=cmap, shading='flat')
        density_unit, temperature_unit = phase['axis_units']
        ax.set(xscale='log', yscale='log', xlabel=f'Gas mass density [{density_unit}]',
               ylabel=f'Gas temperature [{temperature_unit}]')
        ax.figure.colorbar(image, ax=ax, label='Particle count per bin')
        if verbose:
            _print_product_units(phase)
        if show:
            plt.show()
        return ax

    def plot_map(self, image, *, log=None, cmap='magma', ax=None, show=True, verbose=False):
        """Display a prepared map using the current values in its result dictionary.

        Returns: ax (matplotlib.axes.Axes): plotted axes for customization or saving.
        """
        if not isinstance(image, dict) or image.get('product_type') != 'map':
            raise ValueError('Expected a prepared map.')
        ax = plot_map_result(image, ax=ax, log=log, cmap=cmap)
        if verbose:
            _print_product_units(image)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax

    def make_spectrum(self, line, velocity_edges_kms=None, *, wavelength_edges_angstrom=None,
                      rest_wavelength_angstrom=None, los_axis='z', center=None, velocity_frame=None,
                      aperture_radius_kpc=None, periodic=None, chunk_size=100000, verbose=False):
        """Bin line luminosity in velocity or wavelength within a projected aperture.

        Supply exactly one set of bin edges. Wavelength mode uses halo-relative
        velocities and the optical Doppler convention lambda = lambda_0*(1+v/c).
        The rest wavelength can be supplied explicitly or read as a nominal
        value from a label ending in A (angstrom) or m (micrometre).
        Cosmological redshift and thermal/instrumental broadening are omitted.

        Returns: result (dict): luminosity per bin, axis edges, frame and diagnostics.
        """
        if (velocity_edges_kms is None) == (wavelength_edges_angstrom is None):
            raise ValueError('Supply exactly one of velocity_edges_kms or wavelength_edges_angstrom.')
        if wavelength_edges_angstrom is None:
            if rest_wavelength_angstrom is not None:
                raise ValueError('rest_wavelength_angstrom requires wavelength bins.')
            result = self._render(line, [velocity_edges_kms], los_axis, center, velocity_frame,
                                periodic, chunk_size, aperture_radius_kpc, True)
            if verbose:
                _print_product_units(result)
            return result

        edges = _edges(wavelength_edges_angstrom)
        if edges[0] <= 0:
            raise ValueError('Wavelength bin edges must be positive.')
        if velocity_frame is not None and not (isinstance(velocity_frame, str) and velocity_frame == 'halo'):
            raise ValueError("Wavelength mode uses velocity_frame='halo'.")
        wavelength_source = 'explicit'
        if rest_wavelength_angstrom is None:
            match = re.search(r'\s+(\d+(?:\.\d*)?|\.\d+)\s*([Am])\s*$', line)
            if match is None:
                raise ValueError('Supply rest_wavelength_angstrom for a line without a wavelength in its label.')
            rest_wavelength_angstrom = float(match[1]) * (1e4 if match[2] == 'm' else 1.)
            wavelength_source = 'line_label'
        if (isinstance(rest_wavelength_angstrom, (bool, np.bool_))
                or np.ndim(rest_wavelength_angstrom) != 0):
            raise ValueError('Rest wavelength must be a positive finite scalar in angstroms.')
        rest_wavelength_angstrom = float(rest_wavelength_angstrom)
        if not np.isfinite(rest_wavelength_angstrom) or rest_wavelength_angstrom <= 0:
            raise ValueError('Rest wavelength must be positive and finite.')
        result = self._render(line, [edges], los_axis, center, 'halo', periodic, chunk_size,
                              aperture_radius_kpc, True, rest_wavelength_angstrom=rest_wavelength_angstrom)
        result.update(axis_names=['wavelength'], axis_units=['angstrom'],
                      quantity='luminosity per wavelength bin', wavelength_frame='halo',
                      rest_wavelength_angstrom=rest_wavelength_angstrom,
                      rest_wavelength_source=wavelength_source, doppler_convention='optical')
        if verbose:
            _print_product_units(result)
        return result

    def make_radial_profile(self, lines=None, bins=20, *, quantity='luminosity',
                            center=None, periodic=None, chunk_size=100000, verbose=False):
        """Bin retained particles in spherical shells around the chosen center.

        Use an exact label, a sequence of labels, or None for all saved lines.
        Supply nonnegative radial edges in physical kpc or a positive integer
        for equal-width shells spanning zero to the farthest retained particle.
        Luminosity is summed per shell. Cumulative luminosity starts at the first
        edge. Emission is sum(L)/sum(particle volume), with NaN in empty shells.

        Returns: profile (dict): editable values by line, edges, frame and coverage.
        """
        if quantity not in PROFILE_UNITS:
            raise ValueError(f'Choose a radial quantity from {tuple(PROFILE_UNITS)}.')
        labels = self.line_labels if lines is None else (lines,) if isinstance(lines, str) else tuple(lines)
        if not labels or any(label not in self.line_labels for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(f'Choose unique, exact line labels from {self.line_labels}.')
        center, periodic = self._spatial_reference(center, periodic, chunk_size)
        chunks = range(0, self.particle_count, chunk_size)
        with h5py.File(self.filename, 'r') as source:
            positions = source['particles/position_kpc']

            def radii(selection):
                """Calculate spherical radii from one chunk of positions.

                Returns: radius (numpy.ndarray): finite physical radii in kpc.
                """
                offsets = self._position_offsets(positions, selection, center, periodic)
                radius = np.linalg.norm(offsets, axis=1)
                if not np.all(np.isfinite(radius)):
                    raise ValueError('Nonfinite particle radii.')
                return radius

            if isinstance(bins, (int, np.integer)) and not isinstance(bins, (bool, np.bool_)):
                if bins < 1:
                    raise ValueError('The radial bin count must be positive.')
                maximum = max((radii(slice(start, start + chunk_size)).max() for start in chunks), default=0.)
                if maximum == 0:
                    raise ValueError('Provide explicit radial edges when all radii are zero or the catalogue is empty.')
                bins = np.linspace(0., maximum, bins + 1)
            edges = _edges(bins).copy()
            if edges[0] < 0:
                raise ValueError('Radial bin edges must be nonnegative.')
            shell_count = len(edges) - 1
            counts = np.zeros(shell_count, dtype=np.int64)
            volumes = np.zeros(shell_count)
            luminosities = {label: np.zeros(shell_count) for label in labels}
            excluded = dict.fromkeys(labels, 0.)
            datasets = {label: source[f'lines/line_{self.line_labels.index(label):04d}/luminosity_erg_s']
                        for label in labels}
            for start in chunks:
                selection = slice(start, start + chunk_size)
                radius = radii(selection)
                inside = (radius >= edges[0]) & (radius <= edges[-1])
                shell = np.searchsorted(edges, radius[inside], side='right') - 1
                shell[radius[inside] == edges[-1]] = shell_count - 1
                counts += np.bincount(shell, minlength=shell_count)
                volume = source['particles/volume_cm3'][selection]
                if not np.all(np.isfinite(volume)) or np.any(volume <= 0):
                    raise ValueError('Particle volumes must be positive and finite.')
                volumes += np.bincount(shell, weights=volume[inside], minlength=shell_count)
                for label, dataset in datasets.items():
                    luminosity = dataset[selection]
                    if not np.all(np.isfinite(luminosity)) or np.any(luminosity < 0):
                        raise ValueError(f'Invalid particle luminosities for {label}.')
                    luminosities[label] += np.bincount(shell, weights=luminosity[inside], minlength=shell_count)
                    excluded[label] += float(luminosity[~inside].sum())

        included = {label: float(values.sum()) for label, values in luminosities.items()}
        if (not np.all(np.isfinite(volumes)) or not np.all(np.isfinite(list(included.values())))
                or not np.all(np.isfinite(list(excluded.values())))):
            raise ValueError('Nonfinite radial sums (possible numerical overflow).')
        empty = counts == 0
        values = {}
        for label, luminosity in luminosities.items():
            if quantity == 'cumulative_luminosity':
                values[label] = np.cumsum(luminosity)
            elif quantity == 'emission':
                values[label] = np.divide(luminosity, volumes, out=np.full(shell_count, np.nan), where=~empty)
            else:
                values[label] = luminosity
            if np.any(np.isinf(values[label])):
                raise ValueError(f'Nonfinite radial values for {label}.')
        included_count = int(counts.sum())
        result = dict(product_type='radial_profile', quantity=quantity, values=values, lines=labels,
                    edges=[edges], axis_names=['radius'], axis_units=['kpc'], unit=PROFILE_UNITS[quantity],
                    center_kpc=center.copy(), periodic=periodic, particle_count=counts,
                    sampled_volume_cm3=volumes, empty_shell=empty,
                    included_particle_count=included_count,
                    excluded_particle_count=self.particle_count - included_count,
                    included_luminosity_erg_s=included, excluded_luminosity_erg_s=excluded)
        if verbose:
            _print_product_units(result)
        return result

    def make_line_ratio(self, profile, *, numerator, denominator, min_denominator=None, verbose=False):
        """Divide two shell-luminosity curves from a prepared radial profile.

        min_denominator is the minimum accepted denominator luminosity in erg/s.
        Zero denominators and masked inputs produce NaN and a false validity flag.

        Returns: ratio_profile (dict): editable dimensionless curve and shell metadata.
        """
        if (profile.get('product_type') != 'radial_profile' or profile.get('quantity') != 'luminosity'
                or profile.get('unit') != PROFILE_UNITS['luminosity']):
            raise ValueError('Expected a radial profile of luminosity per shell from make_radial_profile().')
        curves = profile['values']
        if numerator not in curves or denominator not in curves:
            raise ValueError('Both line labels must be present in the radial profile.')
        edges = _edges(profile['edges'][0]).copy()
        ratio, valid = _divide_line_values(curves[numerator], curves[denominator],
                                           (len(edges) - 1,), min_denominator)
        label = f'{numerator} / {denominator}'
        result = dict(product_type='radial_profile', quantity='line_ratio', values={label: ratio},
                    lines=(label,), edges=[edges], axis_names=['radius'], axis_units=['kpc'],
                    unit='dimensionless', center_kpc=profile['center_kpc'].copy(),
                    periodic=profile['periodic'], numerator=numerator, denominator=denominator,
                    min_denominator=min_denominator, valid_mask=valid)
        if verbose:
            _print_product_units(result)
        return result

    def make_line_ratio_map(self, numerator_map, denominator_map, *, min_denominator=None, verbose=False):
        """Divide prepared luminosity maps on a shared grid and in the same frame.

        min_denominator is the minimum accepted denominator luminosity per pixel
        in erg/s. Any smoothing must already have been applied to both line maps.

        Returns: ratio_map (dict): editable dimensionless image and validity mask.
        """
        for image in (numerator_map, denominator_map):
            if (image.get('product_type') != 'map' or image.get('quantity') != 'luminosity per pixel'
                    or image.get('unit') != LINE_UNITS['luminosity_erg_s']):
                raise ValueError('Expected luminosity maps from make_map().')
        edges = [_edges(edge) for edge in numerator_map['edges']]
        other_edges = [_edges(edge) for edge in denominator_map['edges']]
        if len(edges) != 2 or len(other_edges) != 2 or any(
                not np.array_equal(a, b) for a, b in zip(edges, other_edges)):
            raise ValueError('Line-ratio maps require identical pixel edges.')
        settings = ('axis_names', 'axis_units', 'center_kpc', 'los_axis', 'periodic',
                    'aperture_radius_kpc', 'smoothing_enabled', 'smoothing_method',
                    'smoothing_sigma_pixels', 'smoothing_kernel_1d', 'smoothing_boundary')
        for key in settings:
            if not np.array_equal(numerator_map[key], denominator_map[key]):
                raise ValueError(f'Line-ratio maps require matching {key}.')
        shape = tuple(len(edge) - 1 for edge in edges)
        ratio, valid = _divide_line_values(numerator_map['values'], denominator_map['values'],
                                           shape, min_denominator)
        numerator = numerator_map['line']
        denominator = denominator_map['line']
        result = {key: deepcopy(numerator_map[key]) for key in settings}
        result.update(product_type='map', quantity='line_ratio', unit='dimensionless',
                      values=ratio, edges=[edge.copy() for edge in edges], valid_mask=valid,
                      line=f'{numerator} / {denominator}', numerator=numerator,
                      denominator=denominator, min_denominator=min_denominator)
        if verbose:
            _print_product_units(result)
        return result

    def plot_profile(self, profile, *, ax=None, show=True, verbose=False):
        """Plot the supplied radial values without reading the particle catalogue.

        Returns: ax (matplotlib.axes.Axes): plotted axes for customization or saving.
        """
        import matplotlib.pyplot as plt

        if profile.get('product_type') != 'radial_profile':
            raise ValueError('Expected a radial profile from make_radial_profile().')
        edges = _edges(profile['edges'][0])
        values = {label: np.asarray(samples, dtype=float) for label, samples in profile['values'].items()}
        if not values or any(samples.shape != (len(edges) - 1,) or np.any(np.isinf(samples))
                             for samples in values.values()):
            raise ValueError('Each radial curve must have one value per shell.')
        if ax is None:
            _, ax = plt.subplots()
        for label, samples in values.items():
            ax.stairs(samples, edges, baseline=None, label=label)
        ylabel = {'luminosity': 'Luminosity per shell', 'cumulative_luminosity': 'Cumulative luminosity',
                  'emission': 'Mean emissivity', 'line_ratio': 'Line ratio'}.get(profile['quantity'], profile['quantity'])
        ax.set(xlabel='Radius [physical kpc]', ylabel=f"{ylabel} [{profile['unit']}]")
        ax.legend()
        if verbose:
            _print_product_units(profile)
        if show:
            plt.show()
        return ax

    def plot_spectrum(self, spectrum, *, ax=None, show=True, verbose=False):
        """Display the supplied velocity or wavelength spectrum.

        Returns: ax (matplotlib.axes.Axes): axes containing the plotted result.
        """
        import matplotlib.pyplot as plt
        if not isinstance(spectrum, dict) or spectrum.get('product_type') != 'spectrum':
            raise ValueError('Expected a prepared spectrum from make_spectrum().')
        edges = _edges(spectrum['edges'][0])
        values = np.asarray(spectrum['values'], dtype=float)
        if values.shape != (len(edges) - 1,) or np.any(np.isinf(values)):
            raise ValueError('The spectrum must have one value per bin.')
        if ax is None:
            _, ax = plt.subplots()
        ax.stairs(values, edges)
        xlabel = 'Wavelength [Å]' if spectrum['axis_names'] == ['wavelength'] else 'LOS velocity [km/s]'
        ax.set(xlabel=xlabel, ylabel='Luminosity per bin [erg/s]')
        ax.set_title(spectrum['line'])
        if verbose:
            _print_product_units(spectrum)
        if show:
            plt.show()
        return ax


def read_EmissionCatalogue(filename, *, verbose=False):
    """Open a particle emission catalogue lazily.

    Set verbose=True to print the catalogue's saved units.

    Returns: catalogue (ParticleEmissionCatalogue): metadata and rendering interface.
    """
    return ParticleEmissionCatalogue(filename, verbose=verbose)
