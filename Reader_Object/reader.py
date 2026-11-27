"""Small read-only interface to codex_simba_line_cubes_v2 HDF5 files.

Opening reads metadata/edges only. Field access returns independent NumPy arrays;
each operation opens/closes its own file, so no persistent handle needs closing.
"""

from pathlib import Path
import json
import warnings

import h5py
import numpy as np


SCHEMA = 'codex_simba_line_cubes_v2'
FIELD_NAMES = {
    'emission': 'volume_emissivity_erg_s_cm3',
    'luminosity': 'cube_erg_s',
    'volume_emissivity_erg_s_cm3': 'volume_emissivity_erg_s_cm3',
    'cube_erg_s': 'cube_erg_s',
}
UNITS = {'cube_erg_s': 'erg s^-1', 'volume_emissivity_erg_s_cm3': 'erg s^-1 cm^-3'}


def _text(value):
    """Decode HDF5 strings without changing line identifiers.

    Returns: text (str): UTF-8 decoded string.
    """
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _check_values(values):
    """Reject nonfinite/negative luminosity or emissivity rather than hiding it.

    Returns: result (NoneType): raises ValueError for invalid values.
    """
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Cube values must be finite and nonnegative.')


def _read_json(group, name, expected_type):
    """Read optional scalar JSON provenance, rejecting malformed stored metadata.

    Returns: value (dict, list or NoneType): decoded metadata; None if not stored
        (or explicitly null where permitted), never invented physical defaults.
    """
    if name not in group:
        return None
    dataset = group[name]
    if not isinstance(dataset, h5py.Dataset) or dataset.shape != ():
        raise ValueError(f'Expected scalar JSON dataset: {dataset.name}')
    try:
        value = json.loads(dataset.asstr()[()])
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f'Invalid JSON metadata: {dataset.name}') from error
    if not isinstance(value, expected_type):
        raise ValueError(f'Unexpected JSON metadata type: {dataset.name}')
    return value


class DataCube:
    """One multi-line cube file; geometry is shared by all stored lines."""

    def __init__(self, filename):
        """Validate file geometry/metadata without loading the 3D data arrays.

        Returns: result (NoneType): initializes the reader or raises an exception.
        """
        self.filename = Path(filename).expanduser().resolve(strict=True)
        with h5py.File(self.filename, 'r') as source:
            schema = _text(source.attrs.get('schema', ''))
            if schema != SCHEMA:
                raise ValueError(
                    f'Expected datacube schema {SCHEMA!r}; found {schema!r}. '
                    'V1 cubes use a different axis order and must not simply be relabelled. '
                    'Emissivity tables are not datacube outputs.'
                )
            if source.attrs.get('complete', False) != True:
                raise ValueError('Datacube is not marked complete.')
            self.metadata = dict(source.attrs)
            self.cube_type = _text(source.attrs['cube_type'])
            expected_order = {'ppp': 'x,y,z', 'ppv': 'x,y,velocity'}
            if self.cube_type not in expected_order:
                raise ValueError(f'Unsupported cube type: {self.cube_type!r}')
            self.axis_order = _text(source.attrs['cube_axis_order'])
            if self.axis_order != expected_order[self.cube_type]:
                raise ValueError(f'Unexpected axis order: {self.axis_order!r}')
            labels = source['lines/labels'].asstr()[...]
            if labels.ndim != 1 or not len(labels):
                raise ValueError('Expected a nonempty one-dimensional line-label list.')
            self.line_labels = tuple(labels.tolist())
            if len(set(self.line_labels)) != len(self.line_labels) or any(not x for x in self.line_labels):
                raise ValueError('Line labels must be nonempty and unique.')
            if source.attrs['line_count'] != len(self.line_labels):
                raise ValueError('line_count does not match the stored labels.')
            if self.cube_type == 'ppp':
                self._edge_names = ('x_edges_kpc', 'y_edges_kpc', 'z_edges_kpc')
                self._image_axes = ('x', 'y')
            else:
                self._edge_names = ('image_x_edges_kpc', 'image_y_edges_kpc', 'velocity_edges_kms')
                self._image_axes = tuple(_text(source.attrs[name]) for name in ('image_x_axis', 'image_y_axis'))
                los = _text(source.attrs['los_axis'])
                if self._image_axes != {'z': ('x', 'y'), 'y': ('x', 'z'), 'x': ('y', 'z')}.get(los):
                    raise ValueError('PPV viewing-direction metadata is inconsistent.')
            self.axes = {}
            for name in self._edge_names:
                dataset = source[f'axes/{name}']
                edges = np.asarray(dataset, dtype=float)
                unit = 'km s^-1' if name == 'velocity_edges_kms' else 'kpc'
                if (edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges))
                        or np.any(np.diff(edges) <= 0) or _text(dataset.attrs['unit']) != unit):
                    raise ValueError(f'Invalid edges or units for {name}.')
                self.axes[name] = edges
            self.shape = tuple(len(self.axes[name]) - 1 for name in self._edge_names)
            if self.cube_type == 'ppp':
                volume = source['geometry/voxel_volume_cm3']
                if (volume.shape != self.shape or _text(volume.attrs['unit']) != 'cm^3'
                        or _text(volume.attrs['axis_order']) != self.axis_order):
                    raise ValueError('PPP voxel-volume geometry must use x,y,z order and match the cube.')
            self.center_kpc = np.asarray(source['geometry/center_kpc'], dtype=float)
            if self.center_kpc.shape != (3,) or not np.all(np.isfinite(self.center_kpc)):
                raise ValueError('center_kpc must contain three finite coordinates.')
            self.provenance = dict(source['provenance'].attrs)
            self.parameters_yaml = source['provenance/parameters_yaml'].asstr()[()]
            self.unit_factors = dict(source['provenance/unit_factors'].attrs)
            # Read only small provenance/coordinate datasets, never table values
            # or 3D cube arrays. Older v2 files can lack these optional records.
            provenance = source['provenance']
            self.unit_inputs = dict(provenance['unit_inputs'].attrs) if 'unit_inputs' in provenance else {}
            self.cosmological_factors = dict(provenance['cosmological_factors'].attrs) if 'cosmological_factors' in provenance else {}
            self.unit_conversions = dict(provenance['unit_conversions'].attrs) if 'unit_conversions' in provenance else {}
            self.table_metadata = {}
            self.table_physics = None
            self.metallicity_mapping = None
            self.table_axis_order = None
            self.table_axes, self.table_axis_metadata = {}, {}
            if 'emissivity_table' in provenance:
                table = provenance['emissivity_table']
                self.table_metadata = dict(table.attrs)
                self.table_physics = _read_json(table, 'physics_commands_json', list)
                self.metallicity_mapping = _read_json(table, 'metallicity_mapping_json', (dict, type(None)))
                if 'axis_order' in table.attrs:
                    self.table_axis_order = tuple(x.strip() for x in _text(table.attrs['axis_order']).split(','))
                    if not all(self.table_axis_order) or len(set(self.table_axis_order)) != len(self.table_axis_order):
                        raise ValueError('Invalid emissivity-table axis order in provenance.')
            if 'table_axes' in provenance:
                table_axes = provenance['table_axes']
                if self.table_axis_order is not None and set(self.table_axis_order) != set(table_axes):
                    raise ValueError('Emissivity-table axis order does not match saved table axes.')
                for name in self.table_axis_order or tuple(table_axes):
                    values = np.asarray(table_axes[name], dtype=float)
                    if (values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values))
                            or np.any(np.diff(values) <= 0)):
                        raise ValueError(f'Invalid saved emissivity-table axis: {name}')
                    self.table_axes[name] = values
                    self.table_axis_metadata[name] = dict(table_axes[name].attrs)
            self.interpolation_method = self.table_metadata.get('interpolation_method')
            self.metallicity_clipping_caveat = self.table_metadata.get('metallicity_clipping_caveat')
            self.line_metadata = {}
            self.clipping_counts = {}
            self.fields = {}
            for index, label in enumerate(self.line_labels):
                group = source[f'lines/line_{index:04d}']
                if _text(group.attrs['line_label']) != label or group.attrs['line_index'] != index:
                    raise ValueError('Line group does not match labels/index order.')
                self.line_metadata[label] = dict(group.attrs)
                counts = None
                if 'clipping_counts_json' in group.attrs:
                    try:
                        counts = json.loads(_text(group.attrs['clipping_counts_json']))
                    except (json.JSONDecodeError, TypeError) as error:
                        raise ValueError(f'Invalid clipping-count JSON for {label}.') from error
                    if not isinstance(counts, dict) or any(
                            not isinstance(v, dict) or set(v) != {'below', 'above'}
                            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in v.values())
                            for v in counts.values()):
                        raise ValueError(f'Invalid clipping-count structure for {label}.')
                self.clipping_counts[label] = counts
                self.fields[label] = tuple(group.keys())
                if 'cube_erg_s' not in group:
                    raise ValueError(f'Missing luminosity cube for {label}.')
                for name, dataset in group.items():
                    if (name not in UNITS or dataset.shape != self.shape
                            or _text(dataset.attrs['unit']) != UNITS[name]
                            or _text(dataset.attrs['axis_order']) != self.axis_order):
                        raise ValueError(f'Unsupported field/shape/units: {label}: {name}')

    def _select_lines(self, lines):
        """Resolve exact labels or zero-based indices; reject duplicate requests.

        Returns: selection (tuple[list[str], bool]): selected labels and single-line flag.
        """
        single = isinstance(lines, (str, int, np.integer))
        requested = self.line_labels if lines is None else ([lines] if single else list(lines))
        selected = []
        for item in requested:
            if isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)):
                if not 0 <= item < len(self.line_labels):
                    raise IndexError(f'Line index {item} is out of range.')
                item = self.line_labels[item]
            if item not in self.line_labels:
                raise KeyError(f'Unknown line {item!r}. Available: {self.line_labels}')
            selected.append(item)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('Select at least one line, without duplicates.')
        return selected, single

    def describe(self):
        """Print geometry, stored quantities, and per-line aperture/clip diagnostics.

        Returns: result (NoneType): prints a human-readable summary.
        """
        print(f'File: {self.filename}\nSchema: {SCHEMA}\nType: {self.cube_type.upper()}')
        print(f'Shape: {self.shape}; array order: {self.axis_order}')
        print(f'Redshift: {self.metadata.get("redshift", "not stored")}')
        for name in self._edge_names:
            edge = self.axes[name]
            print(f'  {name}: {edge[0]:g} to {edge[-1]:g}; {len(edge)-1} bins')
        print('Fields: luminosity = erg s^-1 per voxel; emission = erg s^-1 cm^-3 (when stored).')
        if self.table_metadata:
            print(f'Emissivity table: {self.table_metadata.get("schema", "not stored")}')
        if self.interpolation_method is not None:
            print(f'Interpolation: {self.interpolation_method}')
        for name, values in self.table_axes.items():
            unit = self.table_axis_metadata[name].get('unit', 'unit not stored')
            print(f'  Table {name}: {values[0]:g} to {values[-1]:g}; {len(values)} points [{unit}]')
        if self.metallicity_mapping is not None:
            print(f'Metallicity mapping: {self.metallicity_mapping.get("mapping", "not stored")}')
        if self.metallicity_clipping_caveat is not None:
            print(self.metallicity_clipping_caveat)
        for index, label in enumerate(self.line_labels):
            info = self.line_metadata[label]
            print(f'  [{index}] {label}: {", ".join(self.fields[label])}')
            for name in ('included_luminosity_erg_s', 'excluded_luminosity_erg_s', 'clipped_particle_count',
                         'metallicity_clipped_particle_count', 'zero_metallicity_particle_count',
                         'zero_emissivity_particle_count'):
                if name in info:
                    print(f'      {name}: {info[name]}')
            if self.clipping_counts[label] is not None:
                print(f'      Clipping by axis: {self.clipping_counts[label]}')
        print('No 3D arrays loaded by describe().')

    def get_diagnostics(self, lines=None):
        """Return saved per-line diagnostics and decoded clipping counts.

        Counts concern selected input particles, before aperture exclusion;
        they are not voxel counts or luminosity fractions. No particle Z/X
        arrays are invented from a cube that only stores their mapping/counts.

        Returns: diagnostics (dict): one metadata dictionary for a scalar line
            selector; label-keyed dictionaries for several/omitted lines.
        """
        labels, single = self._select_lines(lines)
        result = {}
        for label in labels:
            info = dict(self.line_metadata[label])
            counts = self.clipping_counts[label]
            info['clipping_counts'] = None if counts is None else {k: dict(v) for k, v in counts.items()}
            total = info.get('input_particle_count')
            if total is not None:
                for name in ('metallicity_clipped_particle_count', 'zero_metallicity_particle_count'):
                    if name in info:
                        count = info[name]
                        if total < 0 or count < 0 or count > total:
                            raise ValueError(f'Invalid particle diagnostic {name} for {label}.')
                        info[name.replace('_particle_count', '_particle_fraction')] = count / total if total else None
            result[label] = info
        return result[labels[0]] if single else result

    def get_fields(self, field, lines=None, *, selection=None):
        """Read emission or luminosity for exact line labels or zero-based indices.

        'emission' means stored volume emissivity, NOT luminosity. PPV files
        normally store luminosity only; unavailable fields raise KeyError.
        Arrays use (x,y,z) for PPP and (image x,image y,velocity) for PPV.
        selection is an optional NumPy/h5py-style slice, e.g. (slice(None),
        slice(None), 20) for one z/velocity layer. No data are clipped,
        transposed, or normalized on reading.
        Multiple/omitted lines return an ordered dictionary, not a stacked cube.

        Returns: data_output (numpy.ndarray or dict[str, numpy.ndarray]): one array
            for a scalar line selector; label-keyed arrays for a sequence or None.
        """
        if field not in FIELD_NAMES:
            raise KeyError(f'Unknown field {field!r}. Choose one of {tuple(FIELD_NAMES)}.')
        labels, single = self._select_lines(lines)
        name = FIELD_NAMES[field]
        data_output = {}
        with h5py.File(self.filename, 'r') as source:
            for label in labels:
                if name not in self.fields[label]:
                    raise KeyError(f'{name} is not stored for {label}; use luminosity for PPV cubes.')
                index = self.line_labels.index(label)
                dataset = source[f'lines/line_{index:04d}/{name}']
                values = np.asarray(dataset[...] if selection is None else dataset[selection])
                _check_values(values)
                data_output[label] = values
        return data_output[labels[0]] if single else data_output

    def get_statistics(self, field='luminosity', lines=None, *, selection=None,
                       percentiles=(5, 50, 95)):
        """Summarize stored voxel values for selected lines and an optional slice.

        Uses the same field aliases, line selectors and slices as get_fields().
        Zeros are included in every reduction; no logarithms, floors or filtering
        are applied. Means/percentiles are equally weighted per selected voxel,
        not particle or volume weighted. std is the population standard deviation
        (ddof=0). Percentiles use NumPy's linear convention. Units for all value
        statistics match the field; fractions are dimensionless.

        Reads one selected line array at a time, including for exact percentiles.
        Only luminosity has an additive total_luminosity_erg_s; summing volume
        emissivity alone is not a physical luminosity. The saved clipping caveat
        describes the whole input sample, not a measured fraction in the slice.
        Empty selections, invalid values or invalid percentile requests raise.

        Returns: statistics (dict): one summary for a scalar line selector;
            label-keyed summaries for several/omitted lines, suitable for display().
        """
        if field not in FIELD_NAMES:
            raise KeyError(f'Unknown field {field!r}. Choose one of {tuple(FIELD_NAMES)}.')
        levels = np.asarray(percentiles, dtype=float)
        if (levels.ndim != 1 or not np.all(np.isfinite(levels))
                or np.any((levels < 0) | (levels > 100))
                or np.unique(levels).size != levels.size):
            raise ValueError('percentiles must be a one-dimensional sequence of unique values from 0 to 100.')
        labels, single = self._select_lines(lines)
        name = FIELD_NAMES[field]
        statistics = {}
        for label in labels:
            values = self.get_fields(field, lines=label, selection=selection)
            if not values.size:
                raise ValueError(f'Cannot calculate statistics for an empty selection: {label}.')
            positive_count = int(np.count_nonzero(values))
            zero_count = int(values.size - positive_count)
            # Reductions use float64 to avoid float32 accumulation overflow/loss.
            values = np.asarray(values, dtype=np.float64)
            summary = {
                'line_label': label,
                'field': name,
                'unit': UNITS[name],
                'selection_shape': values.shape,
                'voxel_count': int(values.size),
                'positive_voxel_count': positive_count,
                'zero_voxel_count': zero_count,
                'positive_voxel_fraction': positive_count / values.size,
                'zero_voxel_fraction': zero_count / values.size,
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'percentiles': dict(zip(levels.tolist(), np.percentile(values, levels).tolist())),
                'metallicity_clipping_caveat': self.metallicity_clipping_caveat
                    or self.line_metadata[label].get('metallicity_clipping_caveat'),
            }
            if name == 'cube_erg_s':
                summary['total_luminosity_erg_s'] = float(np.sum(values))
            reductions = [summary[key] for key in ('min', 'max', 'mean', 'median', 'std')]
            reductions.extend(summary['percentiles'].values())
            if name == 'cube_erg_s':
                reductions.append(summary['total_luminosity_erg_s'])
            if not np.all(np.isfinite(reductions)):
                raise ValueError(f'Nonfinite statistics (possible numerical overflow) for {label}.')
            statistics[label] = summary
        return statistics[labels[0]] if single else statistics

    def get_radial_profile(self, lines=None, bins=20, *, center_kpc=(0., 0., 0.)):
        """Bin PPP voxel luminosities into spherical shells in physical kpc.

        bins is a positive integer (linear edges from zero to the most distant
        voxel center) or explicit nonnegative, strictly increasing edges in kpc.
        center_kpc is in the cube's OFFSET frame, not absolute snapshot coordinates;
        (0,0,0) is the original target center. No periodic wrapping is performed.
        Shells are [left,right), with the final right edge included, as in NumPy.

        Each entire voxel is assigned by its center; there is no fractional voxel
        overlap calculation. Counts and sampled volumes include zero-emission
        voxels. Mean emissivity is sum(luminosity)/sum(stored voxel volume), not an
        unweighted mean or division by an analytic full-sphere volume. Empty shells
        return NaN mean emissivity and an explicit mask; their sums remain zero.
        Shells extending outside the cube are flagged and warn once per call;
        no missing volume/emission correction or surface-brightness conversion
        is applied. cumulative_luminosity_erg_s starts at the first requested edge,
        so it excludes any luminosity below a nonzero first edge. Outside-range
        counts/luminosities are reported separately, never silently discarded.

        Loads shared voxel geometry and one luminosity cube at a time. The saved
        metallicity caveat concerns the input sample, not resolved radial fractions.

        Returns: profiles (dict): one profile dictionary for a scalar line selector;
            label-keyed profiles otherwise. Arrays have one entry per shell,
            except radius_edges_kpc (bins+1); names encode physical units.
        """
        if self.cube_type != 'ppp':
            raise ValueError('Spherical radial profiles require a PPP datacube; PPV has no spatial third axis.')
        labels, single = self._select_lines(lines)
        center = np.asarray(center_kpc, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError('center_kpc must contain three finite offsets in physical kpc.')
        edges_xyz = [self.axes[name] for name in self._edge_names]
        coordinates = [(edge[:-1] + np.diff(edge) / 2) - center[i]
                       for i, edge in enumerate(edges_xyz)]
        radius = np.sqrt(coordinates[0][:, None, None]**2
                         + coordinates[1][None, :, None]**2
                         + coordinates[2][None, None, :]**2)
        if not np.all(np.isfinite(radius)):
            raise ValueError('Nonfinite radial coordinates; check cube edges and center_kpc.')
        if isinstance(bins, (int, np.integer)) and not isinstance(bins, (bool, np.bool_)):
            if bins <= 0:
                raise ValueError('bins must be a positive integer or increasing nonnegative edges in kpc.')
            maximum = float(radius.max())
            if maximum == 0:
                # A single centered voxel still has finite spatial extent.
                maximum = float(np.sqrt(sum(max(abs(edge[0] - center[i]),
                                                abs(edge[-1] - center[i]))**2
                                            for i, edge in enumerate(edges_xyz))))
            radial_edges = np.linspace(0., maximum, int(bins) + 1)
        else:
            radial_edges = np.asarray(bins, dtype=float)
        if (radial_edges.ndim != 1 or radial_edges.size < 2
                or not np.all(np.isfinite(radial_edges)) or radial_edges[0] < 0
                or np.any(np.diff(radial_edges) <= 0)):
            raise ValueError('bins must be a positive integer or increasing nonnegative edges in kpc.')
        inside = (radius >= radial_edges[0]) & (radius <= radial_edges[-1])
        shell_index = np.searchsorted(radial_edges, radius, side='right') - 1
        shell_index[radius == radial_edges[-1]] = radial_edges.size - 2
        selected_index = shell_index[inside]
        shell_count = radial_edges.size - 1
        count = np.bincount(selected_index, minlength=shell_count)
        empty = count == 0
        inscribed_radius = min(min(center[i] - edge[0], edge[-1] - center[i])
                               for i, edge in enumerate(edges_xyz))
        contained = radial_edges[1:] <= inscribed_radius
        if not np.all(contained):
            warnings.warn('Some radial shells extend outside the cube. Profiles use only sampled '
                          'voxels; inspect shell_fully_inside_cube. No missing-volume correction is applied.',
                          RuntimeWarning, stacklevel=2)
        if np.any(empty):
            warnings.warn('Some radial shells contain no voxel centers: mean emissivity is NaN; '
                          'inspect empty_shell. Consider wider bins than the voxel spacing.',
                          RuntimeWarning, stacklevel=2)
        with h5py.File(self.filename, 'r') as source:
            volume = np.asarray(source['geometry/voxel_volume_cm3'], dtype=float)
        if not np.all(np.isfinite(volume)) or np.any(volume <= 0):
            raise ValueError('Voxel volumes must be finite and strictly positive for radial profiles.')
        # Accumulate each shell directly: differencing cumulative histogram sums
        # can erase faint outer emission next to a much brighter inner shell.
        sampled_volume = np.bincount(selected_index, weights=volume[inside], minlength=shell_count)
        if not np.all(np.isfinite(sampled_volume)) or np.any(sampled_volume[~empty] <= 0):
            raise ValueError('Invalid accumulated radial-shell volume (possible numerical overflow).')
        profiles = {}
        for label in labels:
            luminosity = self.get_fields('luminosity', lines=label).astype(np.float64, copy=False)
            shell_luminosity = np.bincount(selected_index, weights=luminosity[inside], minlength=shell_count)
            cumulative = np.cumsum(shell_luminosity)
            excluded_luminosity = float(luminosity[~inside].sum())
            mean_emissivity = np.full(count.shape, np.nan)
            np.divide(shell_luminosity, sampled_volume, out=mean_emissivity, where=~empty)
            if (not np.all(np.isfinite(shell_luminosity)) or np.any(shell_luminosity < 0)
                    or not np.all(np.isfinite(cumulative)) or not np.isfinite(excluded_luminosity)
                    or not np.all(np.isfinite(mean_emissivity[~empty]))):
                raise ValueError(f'Invalid radial statistics (possible numerical overflow) for {label}.')
            profiles[label] = {
                'line_label': label,
                'center_offset_kpc': center.copy(),
                'radius_edges_kpc': radial_edges.copy(),
                'radius_midpoints_kpc': radial_edges[:-1] + np.diff(radial_edges) / 2,
                'voxel_count': count.copy(),
                'sampled_volume_cm3': sampled_volume.copy(),
                'shell_luminosity_erg_s': shell_luminosity,
                'cumulative_luminosity_erg_s': cumulative,
                'mean_emissivity_erg_s_cm3': mean_emissivity,
                'empty_shell': empty.copy(),
                'shell_fully_inside_cube': contained.copy(),
                'outside_range_voxel_count': int(np.count_nonzero(~inside)),
                'outside_range_luminosity_erg_s': excluded_luminosity,
                'metallicity_clipping_caveat': self.metallicity_clipping_caveat
                    or self.line_metadata[label].get('metallicity_clipping_caveat'),
                'method': 'whole voxels assigned by center; mean emissivity = sum(L)/sum(V)',
            }
        return profiles[labels[0]] if single else profiles

    def plot_radial_profile(self, lines=None, bins=20, *, quantity='emission',
                            center_kpc=(0., 0., 0.), log=True, show=True, ax=None):
        """Display PPP spherical profiles, using get_radial_profile unchanged.

        quantity='emission' plots volume-weighted mean emissivity; 'luminosity'
        plots luminosity per shell; 'cumulative_luminosity' sums luminosity from
        the first requested edge. Bins and center use physical kpc in the cube's
        offset frame. Radius points are arithmetic shell midpoints. The underlying
        calculation assigns whole voxels by their centers, without an analytic
        missing-volume correction. These are intrinsic quantities, not observed
        surface brightness. Empty shells are gaps; open circles mark shells that
        extend beyond the cube. Existing radial warnings and clipping caveats
        remain visible. Saved profiles/cube arrays are never altered for plotting.

        log=True uses logarithmic y with zeros masked and reported; an entirely
        nonpositive/empty selection warns and uses linear display. Radius stays
        linear. Multiple lines share one panel/legend. show=True displays directly;
        show=False allows saving/customization. An optional Matplotlib ax reuses
        an existing axes; otherwise a new figure is created.

        Returns: plot_output (tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]):
            the figure and profile axes, after displaying when show=True.
        """
        import matplotlib.pyplot as plt

        quantities = {
            'emission': ('mean_emissivity_erg_s_cm3', r'Mean emissivity [erg s$^{-1}$ cm$^{-3}$]'),
            'luminosity': ('shell_luminosity_erg_s', r'Luminosity per shell [erg s$^{-1}$]'),
            'cumulative_luminosity': ('cumulative_luminosity_erg_s', r'Cumulative luminosity [erg s$^{-1}$]'),
        }
        if quantity not in quantities:
            raise ValueError(f'Unknown radial quantity {quantity!r}; choose from {tuple(quantities)}.')
        labels, _ = self._select_lines(lines)
        profiles = self.get_radial_profile(labels, bins, center_kpc=center_kpc)
        key, ylabel = quantities[quantity]
        positive = any(np.any(profile[key][~profile['empty_shell']] > 0)
                       for profile in profiles.values())
        use_log = bool(log and positive)
        hidden_zeros = sum(int(np.count_nonzero((p[key] == 0) & ~p['empty_shell']))
                           for p in profiles.values()) if use_log else 0
        if log and not positive:
            warnings.warn('Radial profiles have no positive sampled values; using a linear display.',
                          RuntimeWarning, stacklevel=2)
        elif hidden_zeros:
            warnings.warn(f'{hidden_zeros} zero-valued radial points are masked for logarithmic display only.',
                          RuntimeWarning, stacklevel=2)
        affected = [label for label in labels
                    if self.line_metadata[label].get('metallicity_clipped_particle_count', 0) > 0]
        if affected:
            caveat = profiles[affected[0]]['metallicity_clipping_caveat']
            warnings.warn('Radial profiles include metallicity-clipped particles. '
                          + (caveat or 'The physical clipping caveat is not stored.'),
                          RuntimeWarning, stacklevel=2)
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        else:
            fig = ax.figure
        try:
            partial = False
            for label, profile in profiles.items():
                radius = profile['radius_midpoints_kpc']
                values = profile[key].copy()
                values[profile['empty_shell']] = np.nan
                if use_log:
                    values[values == 0] = np.nan
                curve, = ax.plot(radius, values, label=label)
                incomplete = ~profile['shell_fully_inside_cube'] & np.isfinite(values)
                if np.any(incomplete):
                    ax.plot(radius[incomplete], values[incomplete], linestyle='none',
                            marker='o', markerfacecolor='none', color=curve.get_color(),
                            label='_nolegend_')
                    partial = True
            ax.set(xlabel='Radius [physical kpc]', ylabel=ylabel,
                   xscale='linear', yscale='log' if use_log else 'linear')
            title = 'PPP spherical radial profile'
            if quantity == 'cumulative_luminosity':
                title += f' (cumulative from {next(iter(profiles.values()))["radius_edges_kpc"][0]:g} kpc)'
            if partial:
                title += '\nOpen circles: shells extend outside the cube'
            if affected:
                title += '\nIncludes metallicity-clipped particles; see saved caveat'
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=.25)
        except Exception:
            if created:
                plt.close(fig)
            raise
        if show:
            plt.show()
        return fig, ax

    def plot(self, cube_type=None, *, lines=None, log=True, show=True):
        """Plot luminosity summed over z (PPP) or velocity bins (PPV).

        These are luminosity-per-image-pixel maps, NOT observed surface brightness.
        Stored luminosity is per bin, so PPV summation has no extra velocity-bin
        width. Uses true edges for nonuniform bins, reads one plane at a time,
        and masks zeros only for logarithmic display. No source data are altered.
        The third stored axis is summed. Only the resulting 2D image is transposed
        for Matplotlib's (vertical row, horizontal column) display convention.

        Returns: plot_output (tuple[matplotlib.figure.Figure, numpy.ndarray]):
            figure and 1D array of populated panel axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        if cube_type is not None and cube_type != self.cube_type:
            raise ValueError(f'Stored cube type is {self.cube_type!r}, not {cube_type!r}; no conversion is performed.')
        labels, _ = self._select_lines(lines)
        # CAVEAT: plotting cannot undo enriched-boundary treatment of metal-free
        # particles. Surface the saved approximation once per plot, not per line.
        affected = [label for label in labels
                    if self.line_metadata[label].get('metallicity_clipped_particle_count', 0) > 0]
        if affected:
            caveat = self.metallicity_clipping_caveat
            if caveat is None:
                caveat = self.line_metadata[affected[0]].get('metallicity_clipping_caveat')
            warnings.warn('Maps include metallicity-clipped particles. '
                          + (caveat or 'The physical clipping caveat is not stored.'),
                          RuntimeWarning, stacklevel=2)
        columns = min(3, len(labels))
        rows = (len(labels) + columns - 1) // columns
        fig, panels = plt.subplots(rows, columns, squeeze=False,
                                   figsize=(5 * columns, 4 * rows), constrained_layout=True)
        try:
            with h5py.File(self.filename, 'r') as source:
                for panel, label in zip(panels.flat, labels):
                    index = self.line_labels.index(label)
                    dataset = source[f'lines/line_{index:04d}/cube_erg_s']
                    image = np.zeros(self.shape[:2], dtype=np.float64)
                    for plane_index in range(self.shape[2]):
                        plane = dataset[:, :, plane_index]
                        _check_values(plane)
                        image += plane
                    _check_values(image)
                    positive = image[image > 0]
                    norm = None
                    display = image
                    if log and positive.size:
                        low, high = positive.min(), positive.max()
                        if low == high:
                            low, high = low / 10, high * 10
                        norm = LogNorm(vmin=low, vmax=high)
                        display = np.ma.masked_equal(image, 0)
                    elif log:
                        warnings.warn(f'{label}: all-zero map; using a linear display.', RuntimeWarning, stacklevel=2)
                    mesh = panel.pcolormesh(self.axes[self._edge_names[0]],
                                            self.axes[self._edge_names[1]], display.T,
                                            shading='flat', norm=norm)
                    panel.set(xlabel=f'{self._image_axes[0]} offset [kpc]',
                              ylabel=f'{self._image_axes[1]} offset [kpc]', title=label)
                    panel.set_aspect('equal')
                    fig.colorbar(mesh, ax=panel, label='Luminosity per pixel [erg s$^{-1}$]')
            for panel in panels.flat[len(labels):]:
                panel.set_visible(False)
            title = f'{self.cube_type.upper()} luminosity projection'
            if affected:
                title += '\nIncludes metallicity-clipped particles; see saved caveat'
            fig.suptitle(title)
        except Exception:
            plt.close(fig)
            raise
        if show:
            plt.show()
        return fig, panels.ravel()[:len(labels)]


def read_DataCube(hdf5_output_filename):
    """Open a read-only cube interface; this does not load the full cube arrays.

    Returns: datacube (DataCube): validated metadata and on-demand field access.
    """
    return DataCube(hdf5_output_filename)
