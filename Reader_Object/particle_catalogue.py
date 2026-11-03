"""Lazy particle catalogue access and chunked luminosity maps/velocity spectra."""
from pathlib import Path
import json
import h5py
import numpy as np

SCHEMA = 'codex_simba_particle_emission_v1'
PARTICLE_UNITS = dict(snapshot_row='dimensionless', particle_id='dimensionless', position_kpc='kpc',
                      velocity_kms='km s^-1', volume_cm3='cm^3', smoothing_length_kpc='kpc')
LINE_UNITS = {'luminosity_erg_s': 'erg s^-1', 'volume_emissivity_erg_s_cm3': 'erg s^-1 cm^-3'}


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


class ParticleEmissionCatalogue:
    """A read-only catalogue; constructor reads metadata, not particle arrays."""

    def __init__(self, filename):
        """Validate layout and load small metadata.

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
            for name, unit in PARTICLE_UNITS.items():
                shape = (self.particle_count, 3) if name in ('position_kpc', 'velocity_kms') else (self.particle_count,)
                dataset = source['particles/' + name]
                if dataset.shape != shape or dataset.attrs.get('unit') != unit:
                    raise ValueError(f'Invalid particle layout/units: {name}')
                if name in ('particle_id', 'snapshot_row') and not np.issubdtype(dataset.dtype, np.integer):
                    raise ValueError(f'{name} must retain integer precision.')
            reference = source['reference_frame']
            for name in ('center_kpc', 'bulk_velocity_kms', 'halo_center_kpc', 'halo_bulk_velocity_kms', 'box_size_kpc'):
                unit = 'km s^-1' if 'velocity' in name else 'kpc'
                if reference[name].attrs.get('unit') != unit:
                    raise ValueError(f'Invalid reference units: {name}')
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
                self.diagnostics[label] = json.loads(group.attrs['diagnostics_json'])
            self.provenance = dict(source['provenance'].attrs)
            self.table_metadata = json.loads(source['provenance/table_metadata_json'].asstr()[()])
            self.parameters_yaml = source['provenance/parameters_yaml'].asstr()[()]

    def describe(self):
        """Summarize available particles, lines and reference settings.

        Returns: summary (dict): particle count, line labels and physical frame.
        """
        return dict(particle_count=self.particle_count, line_labels=self.line_labels,
                    center_kpc=self.center_kpc.copy(), bulk_velocity_kms=self.bulk_velocity_kms.copy(),
                    box_size_kpc=self.box_size_kpc, periodic=self.periodic)

    def get_fields(self, field, *, line=None, selection=slice(None)):
        """Read one field or a contiguous subset without loading other fields.

        Returns: values (numpy.ndarray): independent selected particle values.
        """
        if field in PARTICLE_UNITS:
            path = 'particles/' + field
        elif field in LINE_UNITS:
            if line not in self.line_labels:
                raise ValueError(f'Choose an exact line label from {self.line_labels}.')
            path = f'lines/line_{self.line_labels.index(line):04d}/' + field
        else:
            raise KeyError(field)
        with h5py.File(self.filename, 'r') as source:
            return source[path][selection]

    def _render(self, line, edges, los_axis, center, bulk_velocity, periodic, chunk_size, aperture_radius_kpc, spectrum):
        """Accumulate luminosity from aligned chunks in a chosen frame.

        Returns: result (dict): bins, frame, units and conservation accounting.
        """
        if line not in self.line_labels or los_axis not in ('x', 'y', 'z'):
            raise ValueError('Choose an available line and LOS x, y or z.')
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer.')
        if isinstance(center, str) and center == 'halo':
            center = self.halo_center_kpc
        center = self.center_kpc if center is None else _vector(center)
        if isinstance(bulk_velocity, str) and bulk_velocity == 'halo':
            bulk_velocity = self.halo_bulk_velocity_kms
        bulk_velocity = self.bulk_velocity_kms if bulk_velocity is None else _vector(bulk_velocity)
        periodic = self.periodic if periodic is None else periodic
        if not isinstance(periodic, (bool, np.bool_)):
            raise ValueError('periodic must be boolean.')
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
            for start in range(0, self.particle_count, chunk_size):
                selection = slice(start, start + chunk_size)
                position = source['particles/position_kpc'][selection] - center
                luminosity = luminosities[selection]
                if not np.all(np.isfinite(position)) or not np.all(np.isfinite(luminosity)) or np.any(luminosity < 0):
                    raise ValueError('Invalid particle coordinates or luminosities.')
                if periodic:
                    position = (position + self.box_size_kpc/2) % self.box_size_kpc - self.box_size_kpc/2
                coordinates = position[:, image_axes]
                keep = np.ones(len(position), dtype=bool)
                if aperture_radius_kpc is not None:
                    keep &= np.linalg.norm(coordinates, axis=1) <= aperture_radius_kpc
                if spectrum:
                    coordinates = source['particles/velocity_kms'][selection, axis:axis+1] - bulk_velocity[axis]
                    if not np.all(np.isfinite(coordinates)):
                        raise ValueError('Invalid particle velocities.')
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
                    quantity='luminosity per velocity bin' if spectrum else 'luminosity per pixel',
                    axis_units=['km s^-1'] if spectrum else ['kpc', 'kpc'],
                    axis_names=['velocity'] if spectrum else ['xyz'[i] for i in image_axes],
                    line=line, los_axis=los_axis, center_kpc=np.array(center), bulk_velocity_kms=np.array(bulk_velocity),
                    periodic=bool(periodic), aperture_radius_kpc=aperture_radius_kpc,
                    included_luminosity_erg_s=float(included_luminosity), excluded_luminosity_erg_s=float(excluded_luminosity),
                    included_particle_count=int(included_count), excluded_particle_count=self.particle_count-int(included_count))

    def make_map(self, line, *, los_axis='z', half_width_kpc=100., bins=128, x_edges_kpc=None,
                 y_edges_kpc=None, center=None, periodic=None, chunk_size=100000):
        """Project all retained gas onto two image axes using point deposition.

        Returns: result (dict): x-first luminosity image, edges and diagnostics.
        """
        if not np.isfinite(half_width_kpc) or half_width_kpc <= 0:
            raise ValueError('half_width_kpc must be positive.')
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError('bins must be a positive integer.')
        default = np.linspace(-half_width_kpc, half_width_kpc, bins+1)
        return self._render(line, [default if x_edges_kpc is None else x_edges_kpc,
                                   default if y_edges_kpc is None else y_edges_kpc],
                            los_axis, center, None, periodic, chunk_size, None, False)

    def make_spectrum(self, line, velocity_edges_kms, *, los_axis='z', center=None,
                      velocity_frame=None, aperture_radius_kpc=None, periodic=None, chunk_size=100000):
        """Bin peculiar LOS velocities, optionally within a projected circular aperture.

        No thermal/instrumental broadening or Hubble flow is added.
        Returns: result (dict): luminosity per velocity bin, edges and diagnostics.
        """
        return self._render(line, [velocity_edges_kms], los_axis, center, velocity_frame,
                            periodic, chunk_size, aperture_radius_kpc, True)

    def plot(self, result, *, ax=None):
        """Plot a rendered map or spectrum without changing its normalization.

        Returns: ax (matplotlib.axes.Axes): axes containing the plotted result.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        if result['values'].ndim == 2:
            mesh = ax.pcolormesh(*result['edges'], result['values'].T, shading='auto')
            ax.figure.colorbar(mesh, ax=ax, label='Luminosity per pixel [erg/s]')
            ax.set(xlabel=result['axis_names'][0]+' [kpc]', ylabel=result['axis_names'][1]+' [kpc]')
        else:
            ax.stairs(result['values'], result['edges'][0])
            ax.set(xlabel='LOS velocity [km/s]', ylabel='Luminosity per bin [erg/s]')
        ax.set_title(result['line'])
        return ax


def read_EmissionCatalogue(filename):
    """Open a particle emission catalogue lazily.

    Returns: catalogue (ParticleEmissionCatalogue): metadata and rendering interface.
    """
    return ParticleEmissionCatalogue(filename)
