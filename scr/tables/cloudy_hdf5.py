"""Interpolate linear Cloudy emissivity on log T/nH[/metal-scale] coordinates.

CAVEAT: zero/out-of-range particle metal scales are clipped to the finite
metal-grid boundaries. Metal-free gas is approximated as enriched gas, NOT
computed as primordial; counts, warnings and saved provenance disclose this.
Table zeros remain exact zeros, without an emissivity floor.
"""

import warnings
import json

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
try:
    from .gass10_reference import get_reference
except ImportError:  # Preserve standalone table-reader use.
    from gass10_reference import get_reference

# Legacy two-axis tables and documented converter tables have distinct formats.
SCHEMA = 'codex_simba_cloudy_mvp_v1'
CONVERTER_SCHEMA = 'codex_simba_cloudy_emissivity_v2'
SUPPORTED_SCHEMAS = (SCHEMA, 'cloudy_hdf5_mvp_v1', CONVERTER_SCHEMA)
AXIS_UNITS = {'log_temperature_K': 'log10(K)', 'log_nH_cm3': 'log10(cm^-3)',
              'log_metal_scale': 'dex'}
INTERPOLATION_METHOD = 'linear emissivity on logarithmic coordinates; no emissivity floor'
METALLICITY_CAVEAT = (
    'CAVEAT: particle metal scales outside the table are clipped for interpolation only. '
    'Z=0 is assigned the lowest NONZERO table metal scale, NOT a primordial calculation; '
    'this can create artificial metal emission and alter H/He emission. '
    'Original particle Z/X and table emissivities are unchanged.')


def read_cloudy_hdf5_table(filename) -> dict:
    """Read the project's Cloudy axes, labels, and linear volume emissivities.

    Accepts legacy T/nH tables and documented T/nH[/metal-scale] tables.
    Axis names/order come from the emissivity dataset, never group iteration.
    New tables additionally require completion, axis definitions and physics
    metadata. Metal-scale tables require the verified GASS10 reference pattern.
    Other extra axes fail explicitly; finite nonnegative emissivities are allowed.

    Returns: table (dict): ordered axes and metadata, labels, physics commands,
        fixed-radiation redshifts, composition description and linear emissivity.
    """
    with h5py.File(filename, 'r') as source:
        schema = source.attrs.get('schema')
        if schema not in SUPPORTED_SCHEMAS:
            raise ValueError(
                f'{filename}: unsupported HDF5 schema {schema!r}; expected one of {SUPPORTED_SCHEMAS!r}.'
            )
        required = ('axes/log_temperature_K', 'axes/log_nH_cm3',
                    'lines/labels', 'lines/emissivity_erg_s_cm3')
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f'{filename}: Cloudy HDF5 table is missing required datasets: {missing}.')
        documented = schema == CONVERTER_SCHEMA
        if documented and source.attrs.get('complete', False) != True:
            raise ValueError('Cloudy converter table is not marked complete.')
        dataset = source['lines/emissivity_erg_s_cm3']
        axis_order = tuple(name.strip() for name in dataset.attrs.get('axis_order', '').split(','))
        if (not axis_order or axis_order[-1] != 'line'
                or len(set(axis_order)) != len(axis_order)):
            raise ValueError('Cloudy emissivity axis_order must name unique physical axes followed by line.')
        axis_names = axis_order[:-1]
        unsupported = set(axis_names) - AXIS_UNITS.keys()
        if unsupported or not {'log_temperature_K', 'log_nH_cm3'} <= set(axis_names):
            raise ValueError(f'Unsupported Cloudy axes {sorted(unsupported)}; currently requires '
                             'T/nH with optional log_metal_scale. Other axes need gas mappings.')
        if set(source['axes'].keys()) != set(axis_names):
            raise ValueError('Cloudy axes group does not match the declared emissivity axis_order.')
        if dataset.attrs.get('unit') != 'erg s^-1 cm^-3':
            raise ValueError('Cloudy HDF5 emissivity unit is unsupported.')
        if documented and (dataset.ndim != len(axis_order) or any(
                dataset.dims[index].label != name
                or len(dataset.dims[index]) != 1
                or dataset.dims[index][0].name != ('/lines/labels' if name == 'line' else '/axes/' + name)
                for index, name in enumerate(axis_order))):
            raise ValueError('Cloudy dimension scales do not match the declared axis_order.')
        axes, axis_metadata = {}, {}
        for name in axis_names:
            coordinate = source['axes/' + name]
            axes[name] = np.asarray(coordinate, dtype=np.float64)
            metadata = dict(coordinate.attrs)
            if (metadata.get('unit', AXIS_UNITS[name]) != AXIS_UNITS[name]
                    or metadata.get('scale', 'log10') != 'log10'):
                raise ValueError(f'Unsupported unit/scale for {name}.')
            if documented and (not all(key in metadata for key in ('unit', 'scale', 'definition'))
                               or not metadata['definition']):
                raise ValueError(f'Unsupported or missing unit/scale/definition for {name}.')
            # These units are defined by the validated legacy format when absent.
            axis_metadata[name] = {'unit': AXIS_UNITS[name], 'scale': 'log10', **metadata}
        if source['lines/labels'].ndim != 1:
            raise ValueError('Cloudy line labels must be a one-dimensional array.')
        labels = list(source['lines/labels'].asstr()[...])
        values = np.asarray(source['lines/emissivity_erg_s_cm3'], dtype=np.float64)
        abundance_model = source.attrs.get('abundance_model')
        if not isinstance(abundance_model, str) or not abundance_model.strip():
            raise ValueError('Cloudy table must record its abundance_model description.')
        physics = []
        if documented:
            if dataset.attrs.get('convention') != 'intrinsic, linear, angle-integrated volume emissivity':
                raise ValueError('Unsupported Cloudy emissivity convention.')
            if 'physics/parsed_commands_json' not in source:
                raise ValueError('Cloudy converter table is missing physics/parsed_commands_json.')
            physics = json.loads(source['physics/parsed_commands_json'].asstr()[()])
            if not isinstance(physics, list) or any(not isinstance(record, dict) for record in physics):
                raise ValueError('Cloudy physics metadata must be a list of command records.')
            if source.attrs.get('axis_order') != dataset.attrs['axis_order']:
                raise ValueError('Cloudy root and emissivity axis_order disagree.')

    for name, axis in axes.items():
        if (axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis))
                or not np.all(np.diff(axis) > 0)):
            raise ValueError(f'{name} axis must be finite and strictly increasing.')
    if not labels or any(not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError('Cloudy HDF5 line labels must be nonempty and unique.')
    if values.shape != tuple(axes[name].size for name in axis_names) + (len(labels),):
        raise ValueError('Cloudy HDF5 emissivity shape does not match its axes.')
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Linear interpolation requires finite nonnegative emissivities; no data repair is applied.')
    metal_reference = get_reference(physics) if 'log_metal_scale' in axes else None
    radiation_redshifts = []
    for record in physics:
        if record.get('kind') == 'radiation':
            redshift = record.get('redshift')
            if (isinstance(redshift, bool) or not isinstance(redshift, (int, float))
                    or not np.isfinite(redshift)):
                raise ValueError('Radiation commands need a finite fixed redshift for this adapter.')
            radiation_redshifts.append(float(redshift))
    return {
        'schema': schema, 'axis_order': axis_names,
        'axes': axes, 'axis_metadata': axis_metadata,
        'physics': physics, 'radiation_redshifts': radiation_redshifts,
        'line_labels': labels,
        'volume_emissivity_erg_s_cm3': values,
        'abundance_model': abundance_model,
        'metal_reference': metal_reference,
        'interpolation_method': INTERPOLATION_METHOD,
    }


def calculate_particle_line_luminosity(gas_fields: dict, *, table_filename: str,
                                       redshift: float, line_label: str) -> dict:
    """Calculate one line through the same interpolation used for multi-line runs.

    Returns: emission (dict): emissivity, luminosity, abundance and clipping data.
    """
    table = read_cloudy_hdf5_table(table_filename)
    return next(iter_particle_line_luminosities(
        gas_fields, table=table, redshift=redshift, line_labels=[line_label],
    ))


def iter_particle_line_luminosities(gas_fields: dict, *, table: dict,
                                    redshift: float, line_labels: list):
    """Reuse one loaded Cloudy table and gas interpolation points for all lines.

    Finite positive T/nH values outside the grid are clipped for interpolation
    only. One warning per affected axis reports particle counts, not line counts.
    Original gas values and particle volumes are unchanged. Metal grids use
    gas_fields Z_mass_fraction and X_mass_fraction, s=(Z/X)/(Z_ref/X_ref).
    CAVEAT: Z=0 is clipped to the lowest enriched model, not primordial physics.
    Legacy two-axis tables keep fixed composition. All Cloudy tables now use
    linear emissivity interpolation, including exact zeros; log diagnostics may
    legitimately be -inf at zero and are not used for interpolation.
    Recorded HM12/CMB redshifts must match the supplied run redshift.

    Returns: emissions (collections.abc.Iterator): one emission dictionary per
        requested label, in input order, without retaining all particle arrays.
    """
    if not np.isfinite(float(redshift)):
        raise ValueError('redshift must be finite.')
    for table_redshift in table.get('radiation_redshifts', []):
        if not np.isclose(float(redshift), table_redshift, rtol=0, atol=1e-6):
            raise ValueError(f'Run redshift {redshift} differs from fixed Cloudy radiation '
                             f'redshift {table_redshift}; choose a matching table.')
    if (not isinstance(line_labels, list) or not line_labels
            or any(not isinstance(label, str) or not label.strip() for label in line_labels)):
        raise ValueError('line_labels must be a nonempty list of exact identifiers.')
    if len(set(line_labels)) != len(line_labels):
        raise ValueError('Repeated line labels requested.')
    missing = [label for label in line_labels if label not in table['line_labels']]
    if missing:
        raise ValueError(f'Lines {missing!r} absent; available: {table["line_labels"]}')
    fields = {}
    for name in ('nH_cm3', 'T_K', 'volume_cm3'):
        if name not in gas_fields:
            raise KeyError(f'Missing Cloudy MVP gas field: {name}')
        values = np.asarray(gas_fields[name], dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f'{name} must be a finite, positive, one-dimensional array.')
        fields[name] = values
    if len({values.size for values in fields.values()}) != 1:
        raise ValueError('Cloudy MVP fields must align by particle.')

    axes = table['axes']
    coordinates = {'log_temperature_K': np.log10(fields['T_K']),
                   'log_nH_cm3': np.log10(fields['nH_cm3'])}
    zero_metallicity_count = 0
    if 'log_metal_scale' in axes:
        for name in ('Z_mass_fraction', 'X_mass_fraction'):
            if name not in gas_fields:
                raise KeyError(f'Missing Cloudy metal-scale gas field: {name}')
            values = np.asarray(gas_fields[name], dtype=np.float64)
            if values.shape != fields['T_K'].shape or not np.all(np.isfinite(values)):
                raise ValueError(f'{name} must be finite and aligned with the other particle arrays.')
            fields[name] = values
        Z, X = fields['Z_mass_fraction'], fields['X_mass_fraction']
        if (np.any(Z < 0) or np.any(Z >= 1) or np.any(X <= 0)
                or np.any(X > 1) or np.any(X + Z > 1 + 1e-12)):
            raise ValueError('Mass fractions require 0<=Z<1, 0<X<=1, and X+Z<=1.')
        reference = table['metal_reference']
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            metal_scale = (Z / X) / reference['metal_to_hydrogen_mass_ratio']
        # CAVEAT: -inf is a temporary coordinate for exactly Z=0. Clipping
        # below replaces it with the lowest NONZERO metal bin, not a primordial
        # spectrum. Do not modify Z, X or epsilon, and do not add an epsilon floor.
        log_scale = np.full(Z.shape, -np.inf)
        np.log10(metal_scale, out=log_scale, where=metal_scale > 0)
        coordinates['log_metal_scale'] = log_scale
        zero_metallicity_count = int(np.count_nonzero(Z == 0))
    clipped = np.zeros(fields['T_K'].size, dtype=bool)
    clipping_counts = {}
    for name, data in coordinates.items():
        axis = axes[name]
        below = data < axis[0]
        above = data > axis[-1]
        clipped |= below | above
        clipping_counts[name] = {'below': int(np.count_nonzero(below)),
                                 'above': int(np.count_nonzero(above))}
        if np.any(below) or np.any(above):
            warnings.warn(
                f'Cloudy HDF5 {name} clipping: {np.count_nonzero(below)} below, '
                f'{np.count_nonzero(above)} above the table range '
                f'[{axis[0]:g}, {axis[-1]:g}]. Using boundary values for '
                'interpolation only; original gas fields are unchanged.',
                RuntimeWarning, stacklevel=2,
            )
        coordinates[name] = np.clip(data, axis[0], axis[-1])
    axis_names = table.get('axis_order', ('log_temperature_K', 'log_nH_cm3'))
    points = np.column_stack([coordinates[name] for name in axis_names])
    if 'log_metal_scale' in axes:
        warnings.warn('Cloudy metal scale uses particle Z/X and verified GASS10 metal/H; '
                      'relative metal ratios and He/H remain the table pattern. '
                      'No additional metallicity scaling is applied.', RuntimeWarning, stacklevel=2)
        if any(clipping_counts['log_metal_scale'].values()):
            warnings.warn(f'{METALLICITY_CAVEAT} Zero-metallicity particles: {zero_metallicity_count}.',
                          RuntimeWarning, stacklevel=2)
    else:
        warnings.warn(f'Cloudy table uses fixed composition: {table["abundance_model"]} '
                      'SIMBA oxygen and total metallicity do not rescale this emissivity.',
                      RuntimeWarning, stacklevel=2)
    for label in line_labels:
        index = table['line_labels'].index(label)
        interpolator = RegularGridInterpolator(
            tuple(axes[name] for name in axis_names),
            table['volume_emissivity_erg_s_cm3'][..., index],
            method='linear', bounds_error=True,
        )
        volume_emissivity = interpolator(points)
        if not np.all(np.isfinite(volume_emissivity)) or np.any(volume_emissivity < 0):
            raise ValueError('Interpolated emissivity must be finite and nonnegative.')
        log_emissivity = np.full(volume_emissivity.shape, -np.inf)
        np.log10(volume_emissivity, out=log_emissivity, where=volume_emissivity > 0)
        yield {
            'line_label': label,
            'log_volume_emissivity': log_emissivity,
            'volume_emissivity_erg_s_cm3': volume_emissivity,
            'line_luminosity_erg_s': volume_emissivity * fields['volume_cm3'],
            'abundance_model': table['abundance_model'],
            'clipped_particle_count': int(np.count_nonzero(clipped)),
            'clipping_counts': clipping_counts,
            'zero_metallicity_particle_count': zero_metallicity_count,
            'metallicity_clipped_particle_count': sum(clipping_counts.get('log_metal_scale', {}).values()),
            'zero_emissivity_particle_count': int(np.count_nonzero(volume_emissivity == 0)),
            'interpolation_method': INTERPOLATION_METHOD,
            'metallicity_mapping': table.get('metal_reference'),
            'metallicity_clipping_caveat': METALLICITY_CAVEAT if 'log_metal_scale' in axes else 'not applicable: fixed-composition table',
        }
