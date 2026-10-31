"""Convert one-zone Cloudy range grids to a documented N-axis HDF5 table.

Explicit .in, .grd, and .emis paths are required; .out is optional. Recognizes
the current SIMBA input, not the entire Cloudy language. Additional varied
commands can be described with axis_definitions (or --axis-definitions JSON).

Storage order is T, nH, metal scale, then other axes sorted by name, with lines
last. Original command order and raw model rows remain in provenance. This new
schema is intentionally separate from the existing two-axis runtime adapter.
Requires NumPy and h5py; does not run Cloudy or interpolate/modify emissivities.

Example (use the real output paths from the same completed run)::

    python hdf5_converter.py --input model.in --grid model.grd \
        --emis model.emis --out model.out --model-order cloudy-c25-index \
        --output new_table.hdf5

Additional axis definitions use a JSON object, for example::

    {"ionization parameter": {"name": "log_U", "unit": "dex", "scale": "log10",
      "grid_header": "^IONI", "definition": "Log10 ionization parameter."}}

An unrecognized physical command remains in physics/commands and
physics/parsed_commands_json as uninterpreted. Original text, hashes, input
axis order, user notes, and the summary are stored under provenance/.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import tempfile
import warnings

import h5py
import numpy as np


SCHEMA = 'codex_simba_cloudy_emissivity_v2'
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
CANONICAL_AXES = ('log_temperature_K', 'log_nH_cm3', 'log_metal_scale')
AXIS_DEFINITIONS = {
    'coronal': {'name': 'log_temperature_K', 'unit': 'log10(K)',
                'scale': 'log10', 'grid_header': r'^CORO',
                'definition': 'Log10 imposed gas temperature in kelvin.'},
    'constant temperature': {'name': 'log_temperature_K', 'unit': 'log10(K)',
                            'scale': 'log10', 'grid_header': r'^CONS',
                            'definition': 'Log10 imposed gas temperature in kelvin.'},
    'hden': {'name': 'log_nH_cm3', 'unit': 'log10(cm^-3)',
             'scale': 'log10', 'grid_header': r'^HDEN',
             'definition': 'Log10 number density of hydrogen nuclei.'},
    'metals': {'name': 'log_metal_scale', 'unit': 'dex',
               'scale': 'log10', 'grid_header': r'^META',
               'definition': 'Log10 scale multiplying reference metal/H number abundances; not metal mass fraction Z.'},
}


def _numbers(command):
    """Extract standalone numbers, excluding quoted paths and names like HM12.

    Returns: values (list[float]): numerical command arguments.
    """
    command = re.sub(r'"[^"]*"|\'[^\']*\'', '', command)
    return [float(value) for value in re.findall(r'(?<![\w.])(' + NUMBER + r')(?![\w.])', command)]


def _line_identity(label):
    """Normalize species spacing and wavelength units for header comparison.

    Returns: identity (tuple): normalized species and wavelength in angstroms.
    """
    match = re.fullmatch(r'\s*(.+?)\s+(' + NUMBER + r')\s*([Am])\s*', label)
    if match is None:
        raise ValueError(f'Unsupported line identifier: {label!r}; use species and A/m wavelength.')
    species, wavelength, unit = match.groups()
    return ' '.join(species.lower().split()), float(wavelength) * (1e4 if unit == 'm' else 1)


def _range_axis(command, descriptor):
    """Read an inclusive range in the coordinate used by the Cloudy grid.

    Returns: values (numpy.ndarray): ascending expected coordinate values.
    """
    match = re.fullmatch(
        r'grid\s+range\s+from\s+(' + NUMBER + r')\s+to\s+(' + NUMBER
        + r')\s+step\s+(' + NUMBER + r')(.*)', command, re.IGNORECASE)
    if match is None:
        raise ValueError('Supported grid syntax: grid range from LOWER to UPPER step STEP [ncpus N].')
    lower, upper, step = map(float, match.groups()[:3])
    options = re.sub(r'\bncpus\s+\d+\b|\bsequential\b|\blog\b|\blinear\b',
                     '', match[4], flags=re.IGNORECASE).strip()
    if options or step == 0 or not np.all(np.isfinite([lower, upper, step])):
        raise ValueError(f'Unsupported or invalid grid range: {command}')
    if re.search(r'\blinear\b', match[4], re.IGNORECASE) and descriptor['scale'] != 'linear':
        raise ValueError('A linear grid for a log-coordinate command needs an explicit coordinate definition; no automatic reinterpretation.')
    intervals = (upper - lower) / step
    if intervals < 0 or not np.isclose(intervals, round(intervals), rtol=0, atol=1e-8):
        raise ValueError('Grid step must reach the endpoint with a nonnegative integer number of intervals.')
    return np.sort(lower + np.arange(round(intervals) + 1) * step)


def _physics_record(command, line_number):
    """Describe explicit supported commands without inferring omitted physics.

    Returns: record (dict): command/source plus any directly parsed fields.
    """
    words = shlex.split(command)
    lower_words = [word.lower() for word in words]
    lower = command.lower()
    values = _numbers(command)
    record = {'command': command, 'input_line': line_number, 'kind': 'uninterpreted'}
    if lower.startswith('abundances '):
        record.update(kind='abundances', reference=words[1],
                      no_grains_on_this_command=bool(re.search(r'\bno\s+grains\b', lower)))
    elif lower.startswith('table hm12') or lower.startswith('cmb'):
        record.update(kind='radiation', component='HM12' if lower.startswith('table') else 'CMB',
                      redshift='varied' if 'vary' in lower_words else (values[0] if values else 'unspecified'))
    elif lower.startswith(('coronal ', 'constant temperature ')):
        record.update(kind='temperature', treatment='imposed', varied=bool(re.search(r'\bvary\b', lower)))
    elif lower.startswith('hden '):
        record.update(kind='hydrogen_density', varied=bool(re.search(r'\bvary\b', lower)))
    elif lower.startswith('metals ') and 'deplete' not in lower:
        record.update(kind='metal_scale', varied=bool(re.search(r'\bvary\b', lower)),
                      definition='Scale of metal/H number abundances; H and He not scaled.')
        if values:
            record['argument'] = values[0]
            record['argument_scale'] = 'log10' if 'log' in lower_words or (values[0] <= 0 and 'linear' not in lower_words) else 'linear'
    elif lower.startswith(('set dr ', 'stop zone ')):
        record.update(kind='geometry', argument=values[0] if values else 'unspecified')
    elif lower.startswith('iterate'):
        record.update(kind='iteration')
    return record


def read_cloudy_input(filename, *, axis_definitions=None):
    """Read supported range axes, requested lines, explicit physics, and notes.

    axis_definitions maps command prefixes to dictionaries containing name,
    unit, scale (log10/linear), grid_header (regex), and definition. Definitions
    describe the actual .grd coordinate convention, not just input arguments.
    Unknown fixed commands are preserved with kind='uninterpreted'. Includes,
    continuations, and other grid syntaxes require a future parser extension.

    Returns: model (dict): input text, axes in command order, lines and metadata.
    """
    filename = Path(filename)
    raw_text = filename.read_text(encoding='utf-8')
    definitions = {**AXIS_DEFINITIONS, **(axis_definitions or {})}
    definitions = {key.lower(): value for key, value in definitions.items()}
    axes, labels, physics, notes, note_lines = [], [], [], [], []
    pending = None
    in_lines = False
    in_notes = False
    notes_seen = False
    grid_save = emis_save = title = None
    lines = raw_text.splitlines(keepends=True)
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip('\r\n')
        if line.strip() == '##### Your Notes #####':
            if notes_seen or in_lines:
                raise ValueError('Use one user-notes block outside the emission-line list.')
            in_notes = notes_seen = True
            continue
        if line.strip() == '##### End Your Notes #####':
            in_notes = False
            continue
        if in_notes and (not line.strip() or line.lstrip().startswith('#')):
            notes.append(raw_line)
            note_lines.append(line_number)
        if not line.strip():
            if any(item.strip() and not item.lstrip().startswith('#') for item in lines[line_number:]):
                raise ValueError(f'Blank input line {line_number} occurs before active commands; use # separators.')
            continue
        if line.lstrip().startswith('#'):
            continue
        in_notes = False  # Later executable commands are never swallowed as notes.
        if line[0].isspace():
            raise ValueError(f'Indented active Cloudy input at line {line_number}.')
        command = shlex.join(shlex.split(line, comments=True))
        # Keep original quoting/spelling for provenance; normalized text is only for matching.
        command = line.split('#', 1)[0].strip() if '#' not in command else command
        lower = command.lower()
        if in_lines:
            if lower == 'end of lines':
                in_lines = False
            else:
                _line_identity(command)
                labels.append(command)
            continue
        if lower.startswith('title '):
            title = command[6:]
            continue
        if re.match(r'save\s+lines?\s+emissivity\b', lower):
            save_options = shlex.split(lower)[4:]
            if (emis_save is not None or 'last' not in save_options
                    or not any(a == 'no' and b == 'hash' for a, b in zip(save_options, save_options[1:]))
                    or set(save_options) - {'last', 'no', 'hash'}):
                raise ValueError('Use one intrinsic save lines emissivity command with last no hash.')
            emis_save = command
            in_lines = True
            continue
        if lower.startswith('save grid '):
            if grid_save is not None:
                raise ValueError('Use one save grid command.')
            grid_save = command
            continue
        if lower.startswith(('init ', 'include ')) or command.endswith('\\'):
            raise ValueError(f'External init/include or continuation needs explicit parser support: {command}')
        if lower.startswith('grid '):
            if pending is None:
                raise ValueError('Each grid range must follow a supported command marked vary.')
            pending['values'] = _range_axis(command, pending)
            pending['grid_command'] = command
            pending['grid_input_line'] = line_number
            axes.append(pending)
            pending = None
            continue
        if pending is not None:
            raise ValueError('A varied command must be followed by its grid range before another active command.')
        physics.append(_physics_record(command, line_number))
        if re.search(r'\bvary\b', lower):
            matches = [key for key in definitions if lower.startswith(key + ' ')]
            if not matches:
                raise ValueError(f'Unknown varied command {command!r}; supply its axis_definitions entry.')
            descriptor = dict(definitions[max(matches, key=len)])
            if (not {'name', 'unit', 'scale', 'grid_header', 'definition'} <= descriptor.keys()
                    or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', descriptor['name'])
                    or descriptor['scale'] not in ('log10', 'linear')):
                raise ValueError('Axis definitions require a safe name, unit, scale, grid_header, and definition.')
            descriptor.update(command=command, input_line=line_number)
            pending = descriptor
    if in_lines or pending is not None:
        raise ValueError('Unclosed emission list or varied command without a range.')
    if not axes or len({axis['name'] for axis in axes}) != len(axes):
        raise ValueError('Expected one or more distinct named physical axes.')
    if not labels or grid_save is None or emis_save is None:
        raise ValueError('Input must contain a grid save and a nonempty emissivity line list.')
    if [record['command'].lower() for record in physics if record['command'].lower().startswith('stop zone ')] != ['stop zone 1']:
        raise ValueError('This converter requires exactly one stop zone 1 command.')
    return {'input_text': raw_text, 'title': title or '', 'axes': axes,
            'requested_lines': labels, 'physics': physics, 'user_notes': ''.join(notes),
            'note_lines': note_lines, 'grid_save': grid_save, 'emis_save': emis_save}


def _read_text_table(filename):
    """Read Cloudy's tab-separated header and records without changing row order.

    Returns: table (tuple): stripped header and record lists.
    """
    with Path(filename).open(encoding='utf-8', newline='') as source:
        rows = list(csv.reader(source, delimiter='\t'))
    if not rows or not rows[0][0].startswith('#'):
        raise ValueError(f'Missing Cloudy column header: {filename}')
    header = [value.strip() for value in rows[0]]
    records = [[value.strip() for value in row] for row in rows[1:] if row]
    if not records or any(len(row) != len(header) or row[0].startswith('#') for row in records):
        raise ValueError(f'Empty/inconsistent table or repeated header: {filename}')
    return header, records


def read_cloudy_outputs(model, grid_filename, emis_filename, *, model_order='sequential'):
    """Validate one-zone outputs and assemble their complete named N-axis tensor.

    cloudy-c25-index explicitly affirms verified C25 merging by model index in
    both raw files. The .emis format has no model IDs: correspondence cannot be
    independently reconstructed from this file alone. Never sort one file.

    Returns: table (dict): stored axes/tensor and unchanged raw model metadata.
    """
    if model_order not in ('sequential', 'cloudy-c25-index'):
        raise ValueError('model_order must be sequential or cloudy-c25-index.')
    header, rows = _read_text_table(grid_filename)
    input_axes = model['axes']
    if (header[:6] != ['#Index', 'Failure?', 'Warnings?', 'Exit code', '#rank', '#seq']
            or header[-1] != 'grid parameter string' or len(header) != 7 + len(input_axes)):
        raise ValueError('Grid header does not match the number of input axes.')
    for axis, column in zip(input_axes, header[6:-1]):
        if not re.match(axis['grid_header'], column, re.IGNORECASE):
            raise ValueError(f'Grid column {column!r} does not match axis {axis["name"]!r}.')
    if any(row[1] != 'F' or row[3] != 'ok' or row[2] not in ('T', 'F') for row in rows):
        raise ValueError('Cloudy grid contains a failed model, non-ok exit, or unknown warning flag.')
    indices = np.array([int(row[0]) for row in rows])
    ranks = np.array([int(row[4]) for row in rows])
    sequence = np.array([int(row[5]) for row in rows])
    if not np.array_equal(indices, np.arange(len(rows))):
        raise ValueError('Raw grid must already be in contiguous increasing model-index order; do not sort it alone.')
    if np.any(ranks < 0) or np.any(sequence < 0) or len(set(zip(ranks, sequence))) != len(rows):
        raise ValueError('Grid rank/sequence pairs must be nonnegative and unique.')
    if model_order == 'sequential' and (np.any(ranks != 0) or not np.array_equal(sequence, indices)):
        raise ValueError('Parallel outputs require explicit model_order=cloudy-c25-index after verifying merging.')
    warning_flags = np.array([row[2] == 'T' for row in rows])
    if warning_flags.any():
        warnings.warn(f'{warning_flags.sum()} Cloudy models have warnings; inspect their .out records.', RuntimeWarning)
    coordinates = np.array([row[6:-1] for row in rows], dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError('Nonfinite Cloudy model coordinates.')
    stored_axes = sorted(input_axes, key=lambda axis: (
        CANONICAL_AXES.index(axis['name']) if axis['name'] in CANONICAL_AXES else len(CANONICAL_AXES), axis['name']))
    shape = tuple(len(axis['values']) for axis in stored_axes)
    if len(rows) != math.prod(shape):
        raise ValueError(f'Incomplete grid: expected {math.prod(shape)} models, found {len(rows)}.')
    storage_indices = []
    for axis in stored_axes:
        column = next(i for i, item in enumerate(input_axes) if item['name'] == axis['name'])
        values = axis['values']
        # Match the six-decimal Cloudy coordinates without changing physical values.
        right = np.minimum(np.searchsorted(values, coordinates[:, column]), len(values) - 1)
        left = np.maximum(right - 1, 0)
        positions = np.where(abs(coordinates[:, column] - values[left])
                             <= abs(coordinates[:, column] - values[right]), left, right)
        if not np.allclose(coordinates[:, column], values[positions], rtol=0, atol=5.1e-7):
            raise ValueError(f'Output coordinates disagree with the requested {axis["name"]} grid.')
        storage_indices.append(positions)
    flat_indices = np.ravel_multi_index(tuple(storage_indices), shape)
    if np.unique(flat_indices).size != len(rows):
        raise ValueError('Duplicate coordinate combinations; the Cartesian grid is incomplete.')
    emis_header, emis_rows = _read_text_table(emis_filename)
    labels = emis_header[1:]
    if emis_header[0] != '#depth' or len(emis_rows) != len(rows):
        raise ValueError('Expected #depth and exactly one .emis row per grid model (one zone, final iteration).')
    if len(labels) != len(model['requested_lines']) or len(set(labels)) != len(labels):
        raise ValueError('Saved line columns do not match the requested line count or contain duplicates.')
    for requested, saved in zip(model['requested_lines'], labels):
        first, second = _line_identity(requested), _line_identity(saved)
        if first[0] != second[0] or not np.isclose(first[1], second[1], rtol=5e-6, atol=1e-6):
            raise ValueError(f'Requested/saved line order differs: {requested!r} versus {saved!r}.')
    raw_values = np.array(emis_rows, dtype=float)
    if not np.isfinite(raw_values).all() or np.any(raw_values < 0):
        raise ValueError('Depths/emissivities must be finite and nonnegative.')
    emissivity = np.empty(shape + (len(labels),), dtype=float)
    emissivity[tuple(storage_indices)] = raw_values[:, 1:]
    return {'axes': stored_axes, 'labels': labels, 'emissivity': emissivity,
            'coordinates': coordinates, 'storage_indices': np.column_stack(storage_indices),
            'model_index': indices, 'rank': ranks, 'sequence': sequence,
            'warning_flag': warning_flags, 'depth_cm': raw_values[:, 0]}


def convert_to_hdf5(input_filename, grid_filename, emis_filename, output_filename, *,
                    out_filename=None, cloudy_version=None, model_order='sequential',
                    axis_definitions=None, resolved_wavelengths=None, line_match_note=''):
    """Convert explicit Cloudy files, validate the round-trip, and publish once.

    Optional resolved wavelengths must come from reviewed Cloudy diagnostics and
    have a source note. An absent .out/version remains explicitly unconfirmed.
    Exact zero emissivities are preserved and flagged as unsuitable for direct
    log interpolation. Existing files are never overwritten.

    Returns: output_path (pathlib.Path): complete new HDF5 table location.
    """
    paths = {'input': Path(input_filename).resolve(), 'grid': Path(grid_filename).resolve(),
             'emissivity': Path(emis_filename).resolve()}
    if out_filename is not None:
        paths['cloudy_output'] = Path(out_filename).resolve()
    output_path = Path(output_filename).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    for source in paths.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    model = read_cloudy_input(paths['input'], axis_definitions=axis_definitions)
    table = read_cloudy_outputs(model, paths['grid'], paths['emissivity'], model_order=model_order)
    out_text = paths['cloudy_output'].read_text(encoding='utf-8') if 'cloudy_output' in paths else ''
    versions = set(re.findall(r'\bCloudy\s+C?(\d{2}\.\d{2}(?:\.\d+)?)\b', out_text, re.IGNORECASE))
    if len(versions) > 1 or (versions and cloudy_version is not None and cloudy_version not in versions):
        raise ValueError('Conflicting Cloudy version information; review the supplied .out.')
    version = next(iter(versions)) if versions else (cloudy_version or 'unconfirmed')
    if model_order == 'cloudy-c25-index' and not version.startswith('25.'):
        raise ValueError('C25 parallel alignment requires .out version evidence or explicit --cloudy-version 25.xx.')
    if resolved_wavelengths is not None:
        resolved_wavelengths = np.asarray(resolved_wavelengths, dtype=float)
        if (not line_match_note or resolved_wavelengths.shape != (len(table['labels']),)
                or not np.isfinite(resolved_wavelengths).all() or np.any(resolved_wavelengths <= 0)):
            raise ValueError('Resolved wavelengths need one positive angstrom value per line and a provenance note.')
    zero_count = int(np.count_nonzero(table['emissivity'] == 0))
    if zero_count:
        warnings.warn(f'Preserving {zero_count} zero emissivities; direct log interpolation is unsupported.', RuntimeWarning)
    input_order = [axis['name'] for axis in model['axes']]
    stored_order = [axis['name'] for axis in table['axes']]
    summary = (f'{model["title"]}\nSchema: {SCHEMA}\nCloudy version: {version}\n'
               f'Input axes: {", ".join(input_order)}\nStored axes: {", ".join(stored_order)}, line\n'
               f'Shape: {table["emissivity"].shape}; models: {len(table["model_index"])}\n'
               f'Warning models: {table["warning_flag"].sum()}; zero emissivities: {zero_count}\n'
               'Physics metadata extracts explicit commands only; user notes are unverified annotations.\n'
               'This schema requires its own runtime reader; do not relabel as the legacy MVP schema.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.cloudy_conversion_', suffix='.hdf5', dir=output_path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    string_type = h5py.string_dtype('utf-8')
    try:
        with h5py.File(temporary_path, 'w') as destination:
            destination.attrs.update(schema=SCHEMA, complete=False, title=model['title'],
                created_utc=datetime.now(timezone.utc).isoformat(), cloudy_version=version,
                cloudy_version_source='cloudy_output' if versions else ('user_argument' if cloudy_version else 'unconfirmed'),
                axis_order=', '.join(stored_order + ['line']), model_order=model_order,
                model_count=len(table['model_index']), line_count=len(table['labels']),
                zero_emissivity_count=zero_count, log_interpolation_ready=(zero_count == 0))
            axes = destination.create_group('axes')
            lines = destination.create_group('lines')
            label_data = lines.create_dataset('labels', data=table['labels'], dtype=string_type)
            label_data.make_scale('line')
            lines.create_dataset('requested_identifiers', data=model['requested_lines'], dtype=string_type)
            lines.attrs['resolution_status'] = ('User-supplied resolved wavelengths with source note.'
                                                if resolved_wavelengths is not None else
                                                'Unconfirmed: header IDs are requested lookups, not proof of resolved transitions; inspect .out.')
            if resolved_wavelengths is not None:
                resolved = lines.create_dataset('resolved_wavelength_A', data=resolved_wavelengths)
                resolved.attrs.update(unit='Angstrom', source=line_match_note)
            emission = lines.create_dataset('emissivity_erg_s_cm3', data=table['emissivity'], compression='gzip', shuffle=True)
            emission.attrs.update(unit='erg s^-1 cm^-3', axis_order=', '.join(stored_order + ['line']),
                convention='intrinsic, linear, angle-integrated volume emissivity',
                convention_source='supported C25 save lines emissivity command; emergent output rejected')
            for index, axis in enumerate(table['axes']):
                coordinate = axes.create_dataset(axis['name'], data=axis['values'])
                coordinate.attrs.update({key: axis[key] for key in ('unit', 'scale', 'definition', 'command', 'input_line', 'grid_command')})
                coordinate.make_scale(axis['name'])
                emission.dims[index].label = axis['name']
                emission.dims[index].attach_scale(coordinate)
            emission.dims[len(stored_order)].label = 'line'
            emission.dims[len(stored_order)].attach_scale(label_data)
            physics = destination.create_group('physics')
            physics.attrs.update(extraction='partial: explicit supported commands, not all Cloudy defaults',
                                 notes_status='User notes are annotations, not verified physical metadata.')
            physics.create_dataset('commands', data=[record['command'] for record in model['physics']], dtype=string_type)
            physics.create_dataset('input_line_numbers', data=[record['input_line'] for record in model['physics']])
            physics.create_dataset('parsed_commands_json', data=json.dumps(model['physics'], indent=2), dtype=string_type)
            models = destination.create_group('models')
            for key in ('model_index', 'rank', 'sequence', 'warning_flag', 'depth_cm', 'coordinates', 'storage_indices'):
                models.create_dataset(key, data=table[key])
            models['coordinates'].attrs['axis_order'] = ', '.join(input_order)
            models['storage_indices'].attrs['axis_order'] = ', '.join(stored_order)
            models['depth_cm'].attrs['unit'] = 'cm'
            provenance = destination.create_group('provenance')
            provenance.create_dataset('summary', data=summary, dtype=string_type)
            provenance.create_dataset('input_axis_order', data=input_order, dtype=string_type)
            provenance.create_dataset('storage_axis_order', data=stored_order, dtype=string_type)
            provenance.create_dataset('user_notes', data=model['user_notes'], dtype=string_type)
            provenance.create_dataset('user_notes_input_lines', data=model['note_lines'], dtype='i8')
            provenance.create_dataset('axis_definitions_json', data=json.dumps(axis_definitions or {}, indent=2), dtype=string_type)
            provenance.attrs.update(grid_save_command=model['grid_save'], emissivity_save_command=model['emis_save'],
                line_match_note=line_match_note, numerical_changes='none: no floor, clipping, rescaling, or normalization',
                alignment_evidence=('Explicit opt-in to C25 model-index merging in both files; .emis contains no IDs.'
                                    if model_order == 'cloudy-c25-index' else 'Sequential rank 0, sequence equals model index; paired raw files required.'))
            sources = provenance.create_group('sources')
            for role, path in paths.items():
                item = sources.create_group(role)
                item.attrs.update(filename=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                item.create_dataset('text', data=path.read_text(encoding='utf-8'), dtype=string_type)
            provenance['input_text'] = sources['input/text']
            destination.attrs['complete'] = True
        with h5py.File(temporary_path, 'r') as check:
            np.testing.assert_array_equal(check['lines/emissivity_erg_s_cm3'][...], table['emissivity'])
            for axis in table['axes']:
                np.testing.assert_array_equal(check['axes/' + axis['name']][...], axis['values'])
            if list(check['lines/labels'].asstr()[...]) != table['labels']:
                raise ValueError('Line labels changed during HDF5 writing.')
        os.link(temporary_path, output_path)  # Atomic publication; fail if destination appeared.
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def main():
    """Convert explicit paths from the command line and print an inspection summary.

    Returns: result (None): the new HDF5 table is saved and its summary printed.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path, help='Actual executed Cloudy .in file.')
    parser.add_argument('--grid', required=True, type=Path, help='Actual .grd/.grid path.')
    parser.add_argument('--emis', required=True, type=Path, help='Actual .emis path.')
    parser.add_argument('--output', required=True, type=Path, help='New HDF5 destination; never overwritten.')
    parser.add_argument('--out', dest='out_filename', type=Path, help='Optional main Cloudy .out log.')
    parser.add_argument('--cloudy-version', help='Explicit version if not recoverable from .out; recorded as user supplied.')
    parser.add_argument('--model-order', choices=('sequential', 'cloudy-c25-index'), default='sequential')
    parser.add_argument('--axis-definitions', type=Path, help='Optional JSON mapping of additional command prefixes to axis definitions.')
    parser.add_argument('--resolved-wavelengths', nargs='+', type=float)
    parser.add_argument('--line-match-note', default='')
    options = vars(parser.parse_args())
    definitions = options.pop('axis_definitions')
    output = convert_to_hdf5(options.pop('input'), options.pop('grid'), options.pop('emis'),
                             options.pop('output'), axis_definitions=json.loads(definitions.read_text()) if definitions else None,
                             **options)
    with h5py.File(output, 'r') as source:
        print(source['provenance/summary'].asstr()[()])
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
