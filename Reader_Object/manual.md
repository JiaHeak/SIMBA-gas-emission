# Manual for Reader_Object

Copy each Python block into a separate Jupyter code cell and run in order.
Use a kernel with NumPy, h5py and Matplotlib. Restart the kernel after updating the reader.

| File | Reader |
| --- | --- |
| Original SIMBA snapshot | `read_RawSnapshot()` |
| Current particle output (`codex_simba_particle_emission_v1`) | `read_EmissionCatalogue()` |
| Existing cube output (`codex_simba_line_cubes_v2`) | `read_DataCube()`; see the last section |

## Cell 1 — Open a particle catalogue

Start Jupyter in `SIMBA-gas-emission/`, or change `PROJECT_DIR` below.
Set `DATA_FILE` to your pipeline output. Opening loads metadata only.

```python
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

PROJECT_DIR = Path.cwd()  # Folder containing scr/ and Reader_Object/
assert (PROJECT_DIR / 'Reader_Object' / '__init__.py').is_file()
sys.path.insert(0, str(PROJECT_DIR))
from Reader_Object import read_EmissionCatalogue

DATA_FILE = PROJECT_DIR / 'particle_emission.hdf5'  # Edit this path
catalogue = read_EmissionCatalogue(DATA_FILE)
display(catalogue.describe())
```

## Cell 2 — Choose a line and inspect particles

Use exact line labels. `selection` reads only the requested rows; omit it to read the full field.

```python
display(list(enumerate(catalogue.line_labels)))
line = catalogue.line_labels[0]  # Change the index
rows = slice(0, 1000)

positions = catalogue.get_fields('position_kpc', selection=rows)  # (N, 3), kpc
epsilon = catalogue.get_fields('volume_emissivity_erg_s_cm3', line=line, selection=rows)
luminosity = catalogue.get_fields('luminosity_erg_s', line=line, selection=rows)
volume = catalogue.get_fields('volume_cm3', selection=rows)

print('Position shape:', positions.shape)
np.testing.assert_allclose(luminosity, epsilon * volume, rtol=1e-12, atol=0)
```

Other fields: `snapshot_row`, `particle_id`, `velocity_kms`, `smoothing_length_kpc`.
Positions are absolute physical box coordinates; velocities are physical peculiar velocities.
`snapshot_row` is the original snapshot row, not a ParticleID.

## Cell 3 — Make a 2D emission map

`make_map()` projects all retained particles along the line of sight.
Choose `field='luminosity'` (default) for erg/s per pixel or
`field='surface_brightness'` for observed line-integrated brightness.
`plot_map()` displays the prepared values with logarithmic colours by default.

```python
image = catalogue.make_map(
    line=line, field='luminosity',
    los_axis='z',
    half_width_kpc=1000,   # Image spans -1000 to +1000 physical kpc
    bins=128,
    center='halo',        # Omit to use the saved default center
    chunk_size=100000,
)
values = image['values']  # Linear NumPy array, image-axis-first
map_ax = catalogue.plot_map(image, cmap='viridis')
```

Change width, bins or viewing direction without rerunning the emission pipeline.
For custom bins, supply `x_edges_kpc` and `y_edges_kpc` for the two image axes.

| `smoothing` in `make_map()` | Kernel |
| --- | --- |
| `False` (default) | Point binning |
| `True` or `'compact'` | [1/8, 3/4, 1/8] per axis, fixed RMS 0.5 pixel |
| `'gaussian'` | Sampled Gaussian with configurable `sigma_pixels`, truncated at four sigma |

For example, `catalogue.make_map(line, smoothing='gaussian', sigma_pixels=1.0)`.
Reflecting boundaries conserve included luminosity. The returned dictionary
records the method, weights, requested width and discrete-kernel RMS
(`smoothing_effective_sigma_pixels`). Widths are in pixels.

Pass `log=False` to `plot_map()` for linear colours. Logarithmic display
masks zeros, and all-zero maps use linear colours with a warning.
Use `show=False` to defer display or `ax=...` to reuse a panel.

```python
brightness = catalogue.make_map(
    line=line, field='surface_brightness', los_axis='z',
    half_width_kpc=1000, bins=128, center='halo',
)
brightness_ax = catalogue.plot_map(brightness, cmap='viridis')
```

Surface brightness divides the binned line luminosity by each physical
pixel area and by `4π`, applies the Tolman factor `(1+z)^-4`, and expresses
the result in erg s⁻¹ cm⁻² arcsec⁻². This assumes isotropic emission and
integrates over the whole line. Optional smoothing acts on luminosity
before conversion. The map records the applied `redshift` and
`pixel_area_kpc2`, while the plot axes remain physical kpc. Pass
`redshift=...` to evaluate dimming at a different redshift for the same
intrinsic emission. The default uses the catalogue redshift.

## Cell 4 — Make a velocity spectrum

The circular aperture is projected on the sky. Velocities are relative to the chosen bulk velocity;
no thermal broadening, instrumental broadening or Hubble flow is added.

```python
spectrum = catalogue.make_spectrum(
    line,
    np.linspace(-300, 300, 61),  # Velocity edges, km/s
    los_axis='z',
    center='halo',
    velocity_frame='halo',      # Omit to use the saved default bulk velocity
    aperture_radius_kpc=200,    # Omit to include all projected radii
)
spectrum_ax = catalogue.plot_spectrum(spectrum)
```

The vertical axis is **erg/s per velocity bin**, not erg/s per km/s.

For a wavelength spectrum, give positive, increasing edges in Å. The Reader
subtracts the halo bulk velocity and applies `lambda = lambda_0 * (1 + v_los/c)`
before binning. The x-axis is in the halo rest frame. A line label ending in
`A` or `m` supplies a nominal rest wavelength (Å or micrometres). Use
`rest_wavelength_angstrom` to specify a more precise value or a line whose label
lacks a wavelength.

```python
# Example for the O VIII line labelled 'O  8 18.9689A'.
wave = catalogue.make_spectrum(
    'O  8 18.9689A',
    wavelength_edges_angstrom=np.linspace(18.94, 19.00, 61),
    center='halo',
    aperture_radius_kpc=200,
)
catalogue.plot_spectrum(wave)
```

For a chosen cosmological redshift `z`, shift the edges afterward. Each value
is intrinsic luminosity integrated over its bin, so this axis transformation
leaves `values` unchanged. To express luminosity per Å, divide by the new bin
widths. Computing observed flux is a separate calculation.

```python
z = 0.6  # Example cosmological redshift
observed = dict(wave, edges=[wave['edges'][0] * (1 + z)], wavelength_frame='observed')
catalogue.plot_spectrum(observed)
```

## Cell 5 — Check coverage and physical assumptions

Rendering processes particles in chunks. Smaller `chunk_size` reduces working memory.
Defaults for center, bulk velocity and periodic wrapping come from the saved reference frame;
explicit centers/velocity frames accept three-component physical vectors.

```python
for name, result in [('Map', image), ('Spectrum', spectrum)]:
    print(name)
    print('Included luminosity [erg/s]:', result['included_luminosity_erg_s'])
    print('Excluded luminosity [erg/s]:', result['excluded_luminosity_erg_s'])
    np.testing.assert_allclose(
        result['values'].sum(), result['included_luminosity_erg_s'], rtol=1e-10, atol=0
    )

display(catalogue.diagnostics[line])
display(catalogue.table_metadata['axes'])
```

- Per-line counts: `input_particle_count`, `computed_particle_count`, `failed_particle_count`,
  `positive_emissivity_particle_count`, and `zero_emissivity_particle_count`.
- Exact zeros are computed results, not failures. Clipped results also count as computed;
  this is numerical completion, not a check of physical accuracy.
- Invalid results abort the job, so published catalogues have zero failed particles.
  Counts are recorded by the writer for new runs; the reader exposes only stored
  diagnostics. Regenerate a catalogue to obtain fields absent from its saved metadata.
- Excluded luminosity reflects the requested map bounds, aperture or velocity range.
- Saved Cloudy diagnostics describe input particles, not individual image pixels.
- Metal-scale clipping, especially assigning an enriched model to Z=0 gas, remains an approximation.
- Maps are intrinsic luminosity, not observed surface brightness or flux.

## Cell 6 — Make and plot a radial profile

Use exact line labels and explicit shell edges in physical kpc. An integer
creates that many equal-width shells from zero to the farthest retained
particle. The Reader uses 3D distances from the chosen center, with the
same periodic wrapping as for maps. Each particle contributes to one shell.

```python
lines = [line]  # Add more exact line labels for comparison.
bins = np.linspace(0, 200, 21)
profile = catalogue.make_radial_profile(lines, bins, quantity='luminosity', center='halo')
profile_ax = catalogue.plot_profile(profile)
```

`profile['values'][line]` contains the editable luminosity per shell in erg/s,
and `profile['edges'][0]` contains the shell boundaries. The profile also
records particle counts, sampled volume and included/excluded luminosities.
For derived curves, update the quantity and unit alongside the values. For
example, divide by the shell widths to show luminosity per kpc of radius.

```python
widths = np.diff(profile['edges'][0])
per_kpc = dict(
    profile,
    values={label: values / widths for label, values in profile['values'].items()},
    quantity='Luminosity per radial interval',
    unit='erg s^-1 kpc^-1',
)
catalogue.plot_profile(per_kpc)
```

`quantity='cumulative_luminosity'` accumulates from the first supplied edge.
`quantity='emission'` gives the volume-weighted mean emissivity of the retained
gas, using its summed particle volumes. Empty shells have zero luminosity and
undefined mean emissivity (NaN).

## Cell 7 — Compare two emission lines

Choose two exact labels from `catalogue.line_labels`. The ratio divides the
two summed luminosities in each shell or pixel. The result is dimensionless.

```python
numerator, denominator = catalogue.line_labels[:2]  # Choose two saved lines.
both_lines = catalogue.make_radial_profile(
    [numerator, denominator], bins, quantity='luminosity', center='halo'
)
ratio_profile = catalogue.make_line_ratio(
    both_lines, numerator=numerator, denominator=denominator
)
catalogue.plot_profile(ratio_profile)

settings = dict(los_axis='z', half_width_kpc=200, bins=128, center='halo', smoothing=True)
numerator_map = catalogue.make_map(numerator, **settings)
denominator_map = catalogue.make_map(denominator, **settings)
ratio_map = catalogue.make_line_ratio_map(numerator_map, denominator_map)
catalogue.plot_map(ratio_map, cmap='viridis', log=False)
```

The map inputs must share pixel edges, frame and smoothing. Build both from
the same catalogue for a matched particle selection. Smoothing is applied
when preparing each luminosity map, before division.

Zero denominators or NaN inputs give NaN in the ratio and `False` in
`valid_mask`. A zero numerator with a positive denominator gives a valid
zero, visible with linear colours. Pass `min_denominator` to exclude values
below a chosen threshold in erg/s per shell or pixel. The plotters draw the
current `values`, so edits made after calculation are visible when replotted.

## Cell 8 — Make a weighted temperature or velocity map

```python
from Reader_Object import read_RawSnapshot

# Set this to the original snapshot if it has moved.
raw_snap = read_RawSnapshot(catalogue.provenance['snapshot_filename'])
temperature = catalogue.make_weighted_map(
    snapshot=raw_snap, los_axis='z', half_width_kpc=200,
    quantity='temperature', weighted_by='mass', bins=128,
)
temperature_ax = catalogue.plot_map(temperature)

velocity = catalogue.make_weighted_map(
    line, los_axis='z', half_width_kpc=200, quantity='velocity',
    weighted_by='luminosity', velocity_frame='halo', bins=128,
)
velocity_ax = catalogue.plot_map(velocity, cmap='coolwarm')
```

Temperature and mass weighting use the original snapshot. The Reader reads
only gas rows inside the image, verifies their particle IDs, and applies the
unit factors saved with the catalogue. Luminosity weighting uses one exact
line label per map. For luminosity-weighted velocity, the catalogue supplies
all fields. Velocity uses the saved bulk velocity by default and plots with
a linear colour scale.

Each pixel contains `sum(weight * quantity) / sum(weight)`. The result
keeps `values`, `weighted_sum`, `weight_sum`, and `valid_mask` for
analysis. Pixels with zero total weight are NaN. With smoothing enabled, the
two sums are smoothed before division.

## Cell 9 — Save a figure

```python
map_ax.figure.savefig(PROJECT_DIR / 'emission_map.png', dpi=200, bbox_inches='tight')
```

## Optional — Read an existing PPP/PPV cube

Run this separately with an existing **cube** file. These methods belong to
`DataCube`. Its voxel-based radial profiles require PPP.

```python
from Reader_Object import read_DataCube

CUBE_FILE = PROJECT_DIR / 'line_cubes.hdf5'  # Edit; not particle_emission.hdf5
datacube = read_DataCube(CUBE_FILE)
datacube.describe()
datacube.plot(lines=0)
display(datacube.get_statistics('luminosity', lines=0))
display(datacube.get_diagnostics(lines=0))

if datacube.cube_type == 'ppp':
    datacube.plot_radial_profile(lines=0, bins=20, quantity='emission')
```
