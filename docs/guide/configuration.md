# Configuration reference

Pass the YAML path as the sole argument to `python scr/pipeline.py`. Filenames in the YAML may be absolute or relative to the YAML file's directory. Keys and identifiers are case-sensitive. Unknown keys raise an error.

## Source

| Setting | Required or default | Meaning |
| --- | --- | --- |
| `type` | `simba` | The supported snapshot format. |
| `snapshot_filename` | Required | Original SIMBA HDF5 snapshot. |
| `catalogue_filename` | Required | Octavius HDF5 catalogue for that snapshot. |
| `redshift` | `auto` | Uses the snapshot redshift. A finite numeric override sets the emission-run redshift and the redshift saved in the output. The snapshot and Octavius catalogue must still agree. |

## Selection

| Setting | Required or default | Meaning |
| --- | --- | --- |
| `group_type` | Required | Use `halo`. |
| `target_index` | Required | Zero-based integer index into the Octavius halo collection. Look it up in the matching catalogue. |

The pipeline starts from that halo's gas membership and keeps gas within `r200m` of `minpot_pos` using minimum-image periodic distances, with hydrogen number density `nH ≤ 0.13 cm⁻³`. `r200m` encloses a mean density of 200 times the cosmic mean matter density.

## Unit factors

| Setting | Required or default | Meaning |
| --- | --- | --- |
| `solar_mass_g` | Required | Grams per solar mass. |
| `kpc_to_cm` | Required | Centimetres per kiloparsec. |
| `specific_energy_to_erg_g` | Required | Converts `(km/s)²` to `erg/g`. |
| `z_sun_mass_fraction` | Required | Reference solar metal mass fraction for `Z/Z☉`. Must be strictly between zero and one. |
| `gamma` | `5/3` | Ratio of specific heats for the gas temperature conversion. Must exceed one. |

Use finite positive values and explicit exponent signs such as `1.0e+10` so PyYAML reads them as numbers. The pipeline computes the scale-factor and Hubble corrections from the snapshot header and uses the Octavius API to obtain catalogue quantities in physical units.

## Emissivity

| Setting | Required or default | Meaning |
| --- | --- | --- |
| `type` | Required | `cloudy_hdf5` or `ps20`. |
| `table_filename` | Required | Path to the matching HDF5 emissivity table. |
| `line_labels` | Required | A nonempty list of unique, exact table identifiers, including internal spaces. This list determines the order of the output lines. |

See [Cloudy table preparation](cloudy.md) for how to list labels from a converted table and [current limitations](limitations.md) for differences between models.

## Reference frame

These settings supply defaults for the Reader. The pipeline saves coordinates in the simulation frame and applies the halo selection described above.

| Setting | Default | Meaning |
| --- | --- | --- |
| `center_kpc` | `auto` | Octavius `minpot_pos`, or a three-component position in physical kpc. |
| `bulk_velocity_kms` | `auto` | Octavius `com_vel_total`, or a three-component physical peculiar velocity in km/s. |
| `periodic` | `true` | Whether the Reader applies periodic wrapping by default. |

## Output

`filename` defaults to `auto`, which writes `particle_emission_halo_<target_index>.hdf5`. Use `.hdf5` or `.h5`; an extensionless name receives `.hdf5`. Use a fresh path for every run because existing files raise `FileExistsError`.

## Example YAML

{download}`Download config.example.yaml <../../config.example.yaml>` and replace the illustrative paths with your inputs.

```{literalinclude} ../../config.example.yaml
:language: yaml
```
