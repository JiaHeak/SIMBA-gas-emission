# Cloudy table preparation

## Prepare the run

Use a completed one-zone Cloudy grid. The converter reads the executed `.in`, the saved grid (`.grd` or `.grid`), and the saved intrinsic line emissivities (`.emis`) from that same run. It accepts an optional main `.out` log for provenance.

- Put `grid range from LOWER to UPPER step STEP` immediately after each command marked `vary`. Choose a step that lands on the upper endpoint. The built-in axis definitions cover temperature (`coronal` or `constant temperature`), hydrogen density (`hden`), and metal scale (`metals`).
- Use `stop zone 1`, one `save grid` command, and one `save lines emissivity "model.emis" last no hash` command followed by your line identifiers and `end of lines`. The saved table must have one emissivity row per model from the final iteration.
- A table for this pipeline needs `log_temperature_K` and `log_nH_cm3`. To vary composition, add the `log_metal_scale` axis using `metals ... log vary` and use `abundances GASS10 no grains` with the reference element pattern and He/H unchanged. A table without the metal axis uses its fixed composition for all particles.
- Match any fixed radiation redshifts recorded for HM12 or the CMB to the run redshift. With `source.redshift: auto`, that is the snapshot redshift.

## Convert the completed grid

```bash
python scr/tables/hdf5_converter.py \
    --input model.in --grid model.grd --emis model.emis \
    --output cloudy_emissivity.hdf5
```

For a serial run, the default `--model-order sequential` requires the grid indices and sequence numbers to be contiguous and ordered. For a parallel Cloudy C25 run, verify that both outputs were merged in the same model order before passing `--model-order cloudy-c25-index`. Supply a version-bearing `--out model.out` or your exact version with `--cloudy-version 25.xx` (replace `25.xx` with the version used).

```{warning}
The `.emis` file has no model IDs. The converter pairs rows in file order, so sorting either input independently can associate emissivities with the wrong gas conditions. Review any Cloudy model warnings in `.out` before using the table.
```

The converter writes a new `codex_simba_cloudy_emissivity_v2` HDF5 file with named axes, line labels, linear angle-integrated volume emissivities in erg s⁻¹ cm⁻³, and input provenance. It rejects incomplete grids, failed models, inconsistent line columns, and negative or nonfinite emissivities.

## Use the table in the pipeline

Set `emissivity.type` to `cloudy_hdf5` and `emissivity.table_filename` to the new HDF5 file. List the exact saved identifiers and their spacing with h5py.

```python
import h5py

with h5py.File('cloudy_emissivity.hdf5', 'r') as table:
    for label in table['lines/labels'].asstr()[:]:
        print(repr(label))
```

Copy the chosen labels into `emissivity.line_labels`. Check the Cloudy log separately to confirm that the saved identifiers refer to the intended transitions. See [current limitations](limitations.md) for the interpolation and abundance assumptions.
