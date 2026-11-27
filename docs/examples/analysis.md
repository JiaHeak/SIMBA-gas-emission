# Analysis examples

## Make a map and spectrum

From the repository root, run the example with your catalogue path.

```bash
python examples/read_particle_catalogue.py particle_emission.hdf5 --width-kpc 200
```

The script prints catalogue metadata and included and excluded luminosity, then plots a projected luminosity map and velocity spectrum. The first saved line is the default. Pass `--line 'O  8 18.9689A'` to choose that exact line when present. The map spans ±200 physical kpc on each image axis and the spectrum uses a 200 kpc projected circular aperture with this command.

## Explore the Reader in Python

The [Reader notebook walkthrough](reader.md) has runnable cells for the methods below. Use the pipeline environment as your notebook kernel and start Jupyter from the checkout root so it can import `Reader_Object`. The walkthrough also covers opening an existing cube.

| Task | Reader methods |
| --- | --- |
| Browse lines, metadata, and selected particle rows | `read_EmissionCatalogue()`, `describe()`, `get_fields()` |
| Make luminosity or surface-brightness maps | `make_map()`, `plot_map()` |
| Bin a velocity or halo-rest-frame wavelength spectrum | `make_spectrum()`, `plot_spectrum()` |
| Compare radial luminosity profiles or line ratios | `make_radial_profile()`, `make_line_ratio()`, `make_line_ratio_map()` |
| Make mass- or luminosity-weighted temperature and velocity maps | `read_RawSnapshot()`, `make_weighted_map()` |
| Follow gas IDs across earlier snapshots | `read_RawSnapshots()`, `track_backward()` in the [gas tracking guide](tracking.md) |
