# Current limitations

## Gas selection

- Each run uses one SIMBA snapshot and one Octavius halo. The fixed halo membership, radius and density cuts are described under [Selection](configuration.md#selection).
- Snapshot arrays must retain the gas row order used to build the Octavius catalogue. Agreement in redshift alone cannot identify the exact snapshot.

## Emission models

| Model | Boundary handling | Composition |
| --- | --- | --- |
| Cloudy HDF5 | Uses linear emissivity interpolation in log temperature and log hydrogen density, with an optional log metal-scale coordinate. Out-of-grid coordinates are clipped to the grid boundary and counted per axis. Exact table zeros are retained. | A two-axis table assumes fixed composition. Metal grids match particle `Z/X` to the verified GASS10 metal/H reference. Individual element ratios and He/H remain fixed to the table pattern. |
| PS20 | Interpolates log emissivity. Temperature, density, and redshift outside the table bounds raise errors. Positive metallicities are clipped to the regular metal grid with a warning. | `Z = 0` uses the primordial slice. Choose a PS20 line table with the intended radiation and shielding model. |

```{warning}
On Cloudy metal grids, `Z = 0` uses the lowest enriched bin and can yield artificial metal-line emission. Check `clipping_counts` and `zero_metallicity_particle_count` in the line diagnostics before interpreting the result.
```

The Cloudy converter can store additional explicitly defined axes, but the runtime currently maps only temperature, hydrogen density, and optional metal scale. A run uses the radiation conditions encoded in its table. It computes intrinsic line emission with no line-of-sight radiative transfer or attenuation step.

## Analysis and tracking

- The current YAML entry point writes particle catalogues. `read_DataCube()` opens existing supported PPV (position–position–velocity) and PPP (position–position–position) cube files.
- Reader surface brightness uses isotropic emission and cosmological dimming. Velocity and wavelength spectra use particle velocities and omit Hubble flow, thermal broadening, and instrumental broadening. See the [analysis examples](../examples/analysis.md) for the available products.
- Gas tracking matches snapshot IDs across epochs. It does not establish particle ancestry, recover type changes, or follow the halo centre over time. See [gas tracking](../examples/tracking.md) for the current duplicate-ID policy and missing matches.
