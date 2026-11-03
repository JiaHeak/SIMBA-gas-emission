"""Reference metal/H mass ratio for the verified Gravity Cloudy C25 GASS10 set.

Number abundances: data/abundances/solar_GASS10.abn, SHA256
f1125b218a81783a7f2c23eb17d210035f0e4fd82bae1f1c5deca00e66907504.
Atomic weights: source/dense.h, SHA256
c8d139bb248cf36ee5f72bfef62958a6001225bf4ff62e5da9359eda06ba3347.
Checked on Gravity 2026-11-02. No unrelated Z_sun constant is substituted.
Only GASS10 without element-specific alterations is supported for metal grids.
"""

import numpy as np

# Atomic-number order H..Zn: (n_element/n_H, Cloudy atomic weight).
REFERENCE = (
    (1., 1.00794), (8.51e-2, 4.0026022), (1.12e-11, 6.9412),
    (2.40e-11, 9.0121823), (5.01e-10, 10.8117), (2.69e-4, 12.01078),
    (6.76e-5, 14.00672), (4.90e-4, 15.99943), (3.63e-8, 18.9984032),
    (8.51e-5, 20.17976), (1.74e-6, 22.989770), (3.98e-5, 24.30506),
    (2.82e-6, 26.9815382), (3.24e-5, 28.08553), (2.57e-7, 30.9737612),
    (1.32e-5, 32.0655), (3.16e-7, 35.4532), (2.51e-6, 39.9481),
    (1.07e-7, 39.09831), (2.19e-6, 40.0784), (1.41e-9, 44.9559108),
    (8.91e-8, 47.8671), (8.51e-9, 50.94151), (4.37e-7, 51.99616),
    (2.69e-7, 54.9380499), (3.16e-5, 55.8472), (9.77e-8, 58.9332009),
    (1.66e-6, 58.69342), (1.55e-8, 63.5463), (3.63e-8, 65.392),
)


def get_reference(physics):
    """Validate GASS10 composition commands and calculate reference mass ratios.

    This matches total metal/H mass, not each particle's individual element
    ratios or He/H. Those remain the table's fixed-pattern approximation.

    Returns: reference (dict): Z_ref, X_ref, Z_ref/X_ref, mapping and provenance.
    """
    abundance = [r for r in physics if r.get('kind') == 'abundances']
    metals = [r for r in physics if r.get('kind') == 'metal_scale']
    if (len(abundance) != 1 or abundance[0].get('reference', '').upper() != 'GASS10'
            or not abundance[0].get('no_grains_on_this_command')
            or len(metals) != 1 or not metals[0].get('varied')
            or metals[0].get('argument_scale') != 'log10'):
        raise ValueError('Metal-axis mapping requires explicit GASS10 no grains and one varied metals command.')
    for record in physics:
        command = record.get('command', '').lower().strip()
        if command.startswith(('element ', 'helium ', 'hydrogen ', 'grains ', 'init ')) or (
                command.startswith('metals ') and record.get('kind') != 'metal_scale'):
            raise ValueError(f'Composition alteration unsupported by GASS10 mapping: {command}')
    numbers = np.asarray(REFERENCE, dtype=np.float64)
    # dense.h stores these literals as float/realnum in the verified build.
    weights = numbers[:, 1].astype(np.float32).astype(np.float64)
    masses = numbers[:, 0] * weights
    metal_mass = float(masses[2:].sum())
    total_mass = float(masses.sum())
    return {
        'reference': 'Cloudy C25 GASS10 H..Zn; fixed solar relative metal pattern',
        'metal_to_hydrogen_mass_ratio': metal_mass / masses[0],
        'Z_ref': metal_mass / total_mass, 'X_ref': float(masses[0] / total_mass),
        'mapping': 's = (gas_fields[Z_mass_fraction] / gas_fields[X_mass_fraction]) / (Z_ref/X_ref)',
        'abundance_source_sha256': 'f1125b218a81783a7f2c23eb17d210035f0e4fd82bae1f1c5deca00e66907504',
        'atomic_weight_source_sha256': 'c8d139bb248cf36ee5f72bfef62958a6001225bf4ff62e5da9359eda06ba3347',
        'approximation': 'Matches total metal/H; individual element ratios and He/H remain the table reference pattern.',
    }
