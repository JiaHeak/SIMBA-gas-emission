"""Learning MVP: .in description -> raw rows -> ordered array -> HDF5.

Keep hdf5_converter.py beside this file: its existing input/text parsers are
reused so the model-to-array mapping stays visible here. Supports the current
one-zone range-grid inputs. This smaller archive has a separate learning schema;
it is not an input format for the current pipeline reader. No emissivity floors,
physical rescaling, or Cloudy execution are performed.
"""

import argparse
import json
from pathlib import Path
import re
import warnings

import h5py
import numpy as np

try:
    from .hdf5_converter import CANONICAL_AXES, read_cloudy_input, _read_text_table, _line_identity
except ImportError:
    from hdf5_converter import CANONICAL_AXES, read_cloudy_input, _read_text_table, _line_identity


def assemble_table(model, grid_filename, emis_filename, *, c25_parallel=False):
    """Pair model rows, then put emissivities into named storage dimensions.

    c25_parallel=True affirms that both C25 files were merged by model index.
    Emissivity rows have no IDs, so that merge convention must be verified.

    Returns: table (dict): ordered axes, line labels, emissivity and raw coordinates.
    """
    # Step 2: read the paired raw files without changing their row order.
    header, rows = _read_text_table(grid_filename)
    emis_header, emis_rows = _read_text_table(emis_filename)
    input_axes = model['axes']
    assert header[:6] == ['#Index', 'Failure?', 'Warnings?', 'Exit code', '#rank', '#seq']
    assert len(header) == len(input_axes) + 7 and header[-1] == 'grid parameter string'
    assert all(re.match(axis['grid_header'], column, re.I)
               for axis, column in zip(input_axes, header[6:-1])), 'Input/grid axis order differs.'
    assert all(row[1] == 'F' and row[3] == 'ok' and row[2] in ('T', 'F') for row in rows), 'Failed Cloudy model.'
    assert [int(row[0]) for row in rows] == list(range(len(rows))), 'Grid must already follow model-index order.'
    execution = [(int(row[4]), int(row[5])) for row in rows]
    assert all(rank >= 0 and seq >= 0 for rank, seq in execution) and len(set(execution)) == len(rows)
    assert c25_parallel or execution == [(0, index) for index in range(len(rows))], 'Verify C25 merging, then enable c25_parallel.'
    warning_count = sum(row[2] == 'T' for row in rows)
    if warning_count:
        warnings.warn(f'{warning_count} models have warnings; inspect the Cloudy .out.', RuntimeWarning)

    labels = emis_header[1:]
    assert emis_header[0] == '#depth' and len(emis_rows) == len(rows), 'Need one final-zone emissivity row per model.'
    assert len(labels) == len(model['requested_lines']) and len(set(labels)) == len(labels)
    for requested, saved in zip(model['requested_lines'], labels):
        species_a, wavelength_a = _line_identity(requested)
        species_b, wavelength_b = _line_identity(saved)
        assert species_a == species_b and np.isclose(wavelength_a, wavelength_b, rtol=5e-6, atol=1e-6), 'Line columns differ.'
    coordinates = np.array([row[6:-1] for row in rows], dtype=float)
    raw_emission = np.array(emis_rows, dtype=float)
    assert np.isfinite(coordinates).all() and np.isfinite(raw_emission).all() and (raw_emission >= 0).all()

    # Step 3: canonical storage order is independent of input command order.
    axes = sorted(input_axes, key=lambda axis: (
        CANONICAL_AXES.index(axis['name']) if axis['name'] in CANONICAL_AXES else len(CANONICAL_AXES), axis['name']))
    shape = tuple(len(axis['values']) for axis in axes)
    assert len(rows) == int(np.prod(shape)), 'Incomplete Cartesian grid.'
    indices = []
    for axis in axes:
        column = next(i for i, item in enumerate(input_axes) if item['name'] == axis['name'])
        # Find each model's coordinate index; allow Cloudy's six-decimal printing.
        nearest = abs(coordinates[:, column, None] - axis['values']).argmin(axis=1)
        assert np.allclose(coordinates[:, column], axis['values'][nearest], rtol=0, atol=5.1e-7), 'Unexpected coordinate.'
        indices.append(nearest)
    assert np.unique(np.ravel_multi_index(tuple(indices), shape)).size == len(rows), 'Repeated coordinate tuple.'
    emissivity = np.empty(shape + (len(labels),))
    emissivity[tuple(indices)] = raw_emission[:, 1:]  # Column zero is depth, not a line.
    return {'axes': axes, 'labels': labels, 'emissivity': emissivity,
            'coordinates': coordinates, 'storage_indices': np.column_stack(indices)}


def convert_mvp(input_filename, grid_filename, emis_filename, output_filename, *, c25_parallel=False):
    """Follow the four conversion steps for the current Cloudy sample.

    Uses assertions for essential learning checks; run without Python's -O flag.
    The full converter adds version evidence, hashes, richer diagnostics, and
    atomic publication. Here complete=False identifies an interrupted write.

    Returns: output_path (pathlib.Path): new HDF5 learning table.
    """
    # Step 1: reuse the input parser for axes, requested lines, physics, and notes.
    model = read_cloudy_input(input_filename)
    table = assemble_table(model, grid_filename, emis_filename, c25_parallel=c25_parallel)
    input_order = [axis['name'] for axis in model['axes']]
    stored_order = [axis['name'] for axis in table['axes']]
    zero_count = int(np.count_nonzero(table['emissivity'] == 0))
    if zero_count:
        warnings.warn(f'Preserving {zero_count} zeros; direct log interpolation is not valid.', RuntimeWarning)

    # Step 4: save arrays and the descriptions needed to understand them.
    output_path = Path(output_filename)
    text_type = h5py.string_dtype('utf-8')
    with h5py.File(output_path, 'x') as destination:  # 'x' refuses to overwrite.
        destination.attrs.update(schema='codex_simba_cloudy_learning_v1', complete=False,
            axis_order=', '.join(stored_order + ['line']), zero_emissivity_count=zero_count,
            model_order='cloudy-c25-index' if c25_parallel else 'sequential')
        axes = destination.create_group('axes')
        for axis in table['axes']:
            coordinate = axes.create_dataset(axis['name'], data=axis['values'])
            coordinate.attrs.update(unit=axis['unit'], definition=axis['definition'])
        lines = destination.create_group('lines')
        lines.create_dataset('labels', data=table['labels'], dtype=text_type)
        lines.create_dataset('requested_identifiers', data=model['requested_lines'], dtype=text_type)
        values = lines.create_dataset('emissivity_erg_s_cm3', data=table['emissivity'])
        values.attrs.update(unit='erg s^-1 cm^-3', axis_order=destination.attrs['axis_order'],
                            convention='intrinsic, linear, angle-integrated volume emissivity')
        models = destination.create_group('models')
        models.create_dataset('coordinates', data=table['coordinates']).attrs['axis_order'] = ', '.join(input_order)
        models.create_dataset('storage_indices', data=table['storage_indices']).attrs['axis_order'] = ', '.join(stored_order)
        physics = destination.create_group('physics')
        physics.create_dataset('parsed_commands_json', data=json.dumps(model['physics'], indent=2), dtype=text_type)
        physics.attrs['interpretation'] = 'Explicit supported commands only; omitted physics is not inferred.'
        provenance = destination.create_group('provenance')
        provenance.create_dataset('input_axis_order', data=input_order, dtype=text_type)
        provenance.create_dataset('user_notes', data=model['user_notes'], dtype=text_type)
        provenance.attrs['user_notes_status'] = 'User annotations, not verified metadata.'
        for role, filename in [('input', input_filename), ('grid', grid_filename), ('emissivity', emis_filename)]:
            raw = provenance.create_dataset(role + '_text', data=Path(filename).read_text(), dtype=text_type)
            raw.attrs['filename'] = str(Path(filename).resolve())
        destination.attrs['complete'] = True
    return output_path


def main():
    """Read four explicit paths and execute the learning converter.

    Returns: result (None): the output path is printed after conversion.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'grid', 'emis', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--c25-parallel', action='store_true', help='Affirm verified C25 model-index merging in both files.')
    args = parser.parse_args()
    print(convert_mvp(args.input, args.grid, args.emis, args.output, c25_parallel=args.c25_parallel))


if __name__ == '__main__':
    main()
