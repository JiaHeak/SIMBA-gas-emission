"""Gas particle ID matching across complete SIMBA snapshots, with lazy fields.

This follows unchanged IDs, not particle ancestry or halo membership. Input
files must be complete, unsliced snapshots from the same simulation run.
"""
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from scr.simba import get_gas_fields
from .raw_snapshot import RawSnapshot


def _integer_vector(values, name):
    """Validate nonnegative integers without a floating-point intermediate.

    Returns: array (numpy.ndarray): validated one-dimensional integer array.
    """
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in 'iu' or np.any(array < 0):
        raise ValueError(f'{name} must be a one-dimensional nonnegative integer array.')
    return array


def _readonly(values):
    """Protect stored matches from accidental in-place changes.

    Returns: array (numpy.ndarray): independent read-only array.
    """
    array = np.array(values, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class TrackedFields:
    """Physical fields aligned to the starting selection, with missing data masked."""

    values: dict
    found: np.ndarray
    snapshot: Path


class ParticleTracks:
    """Matched gas rows indexed by snapshot basename and original selection order.

    Missing rows are -1. Never pass these directly to a raw reader: use
    get_fields(), which reads only found rows and returns masked arrays.
    """

    def __init__(self, collection, current_snapshot, particle_ids, rows):
        """Store the original IDs and the aligned source/target matches.

        Returns: result (NoneType): initializes the tracking result.
        """
        self._collection = collection
        self.current_snapshot = current_snapshot
        self.particle_ids = _readonly(particle_ids)
        self.rows = {key: _readonly(value) for key, value in rows.items()}
        self.found = {key: _readonly(value >= 0) for key, value in self.rows.items()}

    def describe(self):
        """Print and return matching counts for the original selected population.

        Returns: summary (dict[str, dict]): selected, found, missing and scale factor per snapshot.
        """
        summary = {}
        for key, mask in self.found.items():
            summary[key] = dict(selected=len(mask), found=int(mask.sum()),
                                missing=int((~mask).sum()),
                                scale_factor=self._collection._snapshots[key].scale_factor)
            print(f'{key}: {mask.sum():,}/{len(mask):,} gas particles found; {(~mask).sum():,} missing')
        return summary

    def get_fields(self, *, snapshot, fields, chunk_size=100000):
        """Read physical fields without indexing any missing-row sentinel.

        Missing means absent from PartType0, not necessarily physically
        destroyed. Vector fields mask all components for missing particles.

        Returns: fields_output (TrackedFields): masked values and explicit found mask in source order.
        """
        key = self._collection._resolve(snapshot)
        if key not in self.rows:
            raise ValueError(f'{key} was not included in this tracking result.')
        fields = [fields] if isinstance(fields, str) else list(fields)
        allowed = {'T_K', 'mass_g', 'position_kpc', 'velocity_kms'}
        if not fields or set(fields) - allowed or len(set(fields)) != len(fields):
            raise ValueError(f'Choose unique fields from {sorted(allowed)}.')
        if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer.')
        mask = self.found[key]
        values = {field: np.ma.masked_all((len(mask), 3) if field in {'position_kpc', 'velocity_kms'}
                                        else (len(mask),), dtype=float) for field in fields}
        indices = np.flatnonzero(mask)
        for start in range(0, len(indices), chunk_size):
            selected = indices[start:start + chunk_size]
            raw = self._collection._snapshots[key].read_gas_fields(
                self.rows[key][selected], fields=fields, particle_ids=self.particle_ids[selected])
            for field in fields:
                values[field][selected] = raw[field]
        return TrackedFields(values=values, found=_readonly(mask),
                             snapshot=self._collection._snapshots[key].filename)


class TrackSnapshots:
    """Manage single-file snapshots and cache sorted gas-ID lookup arrays.

    Construction reads headers and dataset shapes only. First matching reads
    the entire gas ParticleIDs column of each participating snapshot. Cached
    sorted uint64 IDs and int64 rows cost approximately 16 bytes per gas
    particle per snapshot, plus temporary sorting memory. clear_cache()
    releases these indexes without invalidating previously returned tracks.
    """

    def __init__(self, files, *, unit_inputs=None):
        """Register complete snapshots in caller order and validate shared metadata.

        Returns: result (NoneType): initializes the snapshot collection without loading IDs.
        """
        if isinstance(files, (str, Path)):
            raise ValueError('files must be a sequence of snapshot paths.')
        self.files = tuple(Path(path).expanduser().resolve() for path in files)
        if not self.files or len({path.name for path in self.files}) != len(self.files):
            raise ValueError('Provide at least one snapshot, with unique basenames.')
        self._snapshots = {}
        self._counts = {}
        self._cache = {}
        metadata = []
        for path in self.files:
            snapshot = RawSnapshot(path, unit_inputs=unit_inputs)
            with h5py.File(path, 'r') as source:
                header = source['Header'].attrs
                if int(header.get('NumFilesPerSnapshot', 1)) != 1:
                    raise ValueError(f'{path.name}: split snapshot files are not supported yet.')
                ids = source.get('PartType0/ParticleIDs')
                if ids is None:
                    count = header.get('NumPart_ThisFile')
                    if count is None or int(count[0]) != 0:
                        raise ValueError(f'{path.name}: missing gas ParticleIDs without a zero gas count.')
                    n = 0
                else:
                    if ids.ndim != 1 or ids.dtype.kind not in 'iu':
                        raise ValueError(f'{path.name}: gas ParticleIDs must be a one-dimensional integer dataset.')
                    n = len(ids)
                if 'NumPart_ThisFile' in header and int(header['NumPart_ThisFile'][0]) != n:
                    raise ValueError(f'{path.name}: header gas count disagrees with ParticleIDs.')
                metadata.append({key: np.asarray(header[key]) for key in
                                 ('HubbleParam', 'BoxSize', 'Omega0', 'OmegaLambda') if key in header})
            self._snapshots[path.name] = snapshot
            self._counts[path.name] = n
        for key in ('HubbleParam', 'BoxSize', 'Omega0', 'OmegaLambda'):
            present = [key in item for item in metadata]
            if any(present) and not all(present):
                raise ValueError(f'Snapshots have inconsistent availability of {key}.')
            if all(present) and any(not np.allclose(item[key], metadata[0][key], rtol=1.e-8, atol=0)
                                    for item in metadata[1:]):
                raise ValueError(f'Snapshots disagree on {key}; check they belong to the same run.')

    def _resolve(self, snapshot):
        """Resolve an exact registered path or an unambiguous basename.

        Returns: key (str): snapshot basename used in result dictionaries.
        """
        path = Path(snapshot).expanduser()
        if str(path) == path.name and path.name in self._snapshots:
            return path.name
        path = path.resolve()
        if path in self.files:
            return path.name
        raise KeyError(f'Snapshot is not registered: {snapshot}')

    def _index(self, key):
        """Build an exact integer ID lookup retaining all rows.

        Returns: index (tuple[numpy.ndarray, numpy.ndarray]): sorted uint64 IDs and original rows.
        """
        if key not in self._cache:
            with h5py.File(self._snapshots[key].filename, 'r') as source:
                ids = (source['PartType0/ParticleIDs'][:] if self._counts[key]
                       else np.empty(0, dtype=np.uint64))
            ids = _integer_vector(ids, 'ParticleIDs').astype(np.uint64, copy=False)
            order = np.argsort(ids)
            self._cache[key] = ids[order], order.astype(np.int64, copy=False)
        return self._cache[key]

    def _match_rows(self, key, ids):
        """Look up selected IDs and report any with multiple gas rows.

        Returns: rows (numpy.ndarray): snapshot rows in selection order, or -1 for absent IDs.
        """
        rows = np.full(len(ids), -1, dtype=np.int64)
        if not len(ids):
            return rows
        sorted_ids, sorted_rows = self._index(key)
        first = np.searchsorted(sorted_ids, ids, side='left')
        count = np.searchsorted(sorted_ids, ids, side='right') - first
        ambiguous = np.unique(ids[count > 1])
        if ambiguous.size:
            examples = ', '.join(str(pid) for pid in ambiguous[:5])
            raise ValueError(f'{key}: duplicate gas ParticleIDs make tracking ambiguous for '
                             f'{ambiguous.size:,} selected ID(s). Examples: {examples}.')
        found = count == 1
        rows[found] = sorted_rows[first[found]]
        return rows

    def clear_cache(self):
        """Release cached full-snapshot ID indexes.

        Returns: result (NoneType): clears lookup arrays; existing tracks remain usable.
        """
        self._cache.clear()

    def track_backward(self, *, selected_rows, current_snapshot, target_snapshots,
                       ptype='gas', expected_particle_ids=None):
        """Match the original gas IDs independently in every earlier snapshot.

        selected_rows are PartType0 rows in current_snapshot, not catalogue
        rows or IDs. Missing IDs stay in the result with row -1 and found=False.
        Each selected ID must occur at most once in each participating snapshot.
        Header times enforce direction, irrespective of filename/input order.

        Returns: tracks (ParticleTracks): source and target rows aligned to the starting selection.
        """
        if ptype != 'gas':
            raise ValueError("Only ptype='gas' is supported in this version.")
        current = self._resolve(current_snapshot)
        rows = _integer_vector(selected_rows, 'selected_rows')
        if np.any(rows >= self._counts[current]):
            raise IndexError('A selected row is outside the starting snapshot gas array.')
        rows = rows.astype(np.int64, copy=False)
        if len(np.unique(rows)) != len(rows):
            raise ValueError('Repeated selected rows are not allowed.')
        if isinstance(target_snapshots, (str, Path)):
            raise ValueError('target_snapshots must be a sequence of registered snapshots.')
        targets = [self._resolve(path) for path in target_snapshots]
        if not targets or len(set(targets)) != len(targets):
            raise ValueError('Choose at least one target snapshot without duplicates.')
        for target in targets:
            if self._snapshots[target].scale_factor >= self._snapshots[current].scale_factor:
                raise ValueError(f'{target} is not earlier than {current} according to Header/Time.')
        ids = (get_gas_fields(self._snapshots[current].filename, rows, [])['PartType0/ParticleIDs']
               if len(rows) else np.empty(0, dtype=np.uint64))
        ids = _integer_vector(ids, 'ParticleIDs').astype(np.uint64, copy=False)
        if expected_particle_ids is not None:
            expected = _integer_vector(expected_particle_ids, 'expected_particle_ids').astype(np.uint64, copy=False)
            if not np.array_equal(ids, expected):
                raise ValueError('Starting snapshot rows do not match the catalogue ParticleIDs.')
        self._match_rows(current, ids)
        matched = {current: rows}
        for target in targets:
            matched[target] = self._match_rows(target, ids)
        return ParticleTracks(self, current, ids, matched)


def read_RawSnapshots(files, *, unit_inputs=None):
    """Open a collection for gas tracking with optional pipeline unit inputs.

    Returns: snapshots (TrackSnapshots): header-only collection until matching or field reads.
    """
    return TrackSnapshots(files, unit_inputs=unit_inputs)
