# Output format

Each run writes one `codex_simba_particle_emission_v1` HDF5 catalogue. The root attributes record `schema`, `complete`, `particle_count`, `line_count`, and the emission run `redshift`. Each line's particle arrays align with the same rows in `particles/`.

## Particle fields

| Dataset under `particles/` | Shape | Meaning and units |
| --- | --- | --- |
| `snapshot_row` | `(N,)` | Integer gas row index into `PartType0` in the original snapshot. |
| `particle_id` | `(N,)` | Integer SIMBA `ParticleIDs` value. IDs can repeat. |
| `position_kpc` | `(N, 3)` | Physical simulation-box position in kpc, ordered x, y, z. |
| `velocity_kms` | `(N, 3)` | Physical peculiar velocity in km/s, ordered x, y, z. |
| `volume_cm3` | `(N,)` | Particle volume from mass divided by mass density, in cm³. |
| `smoothing_length_kpc` | `(N,)` | Physical GIZMO kernel support radius in kpc. |

## Line fields

`lines/labels` contains the exact labels in configuration order. Each `lines/line_0000`, `lines/line_0001`, and subsequent group records its `line_label` and `diagnostics_json` attributes.

| Dataset in each line group | Shape | Units |
| --- | --- | --- |
| `volume_emissivity_erg_s_cm3` | `(N,)` | erg s⁻¹ cm⁻³ |
| `luminosity_erg_s` | `(N,)` | erg s⁻¹ |

The per-particle luminosity is `L = ε × V`, where `ε` is the angle-integrated volume emissivity and `V` is the particle volume. The writer validates the relation and rejects invalid values before publishing the file. Inspect stored counts and model diagnostics with `catalogue.diagnostics[line]` after [opening the catalogue](../examples/analysis.md). Exact zeros and clipped table evaluations both count as computed values.

## Frame and provenance

| Group | Contents |
| --- | --- |
| `reference_frame/` | Reader defaults `center_kpc` and `bulk_velocity_kms`, the original `halo_center_kpc` and `halo_bulk_velocity_kms`, `box_size_kpc`, and the `periodic` attribute. |
| `provenance/` | Resolved settings in `parameters_yaml`, conversion factors in `unit_factors_json`, and `table_metadata_json`. Attributes identify the input files, halo index, model, and selection. |

Open this schema with `Reader_Object.read_EmissionCatalogue()`. For temperature, density, or mass-dependent analysis, keep the original snapshot available. For existing PPP or PPV cube files, see the cube example in the [Reader notebook walkthrough](../examples/reader.md).
