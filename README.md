# Pipeline for SIMBA Line Emission Data Cube
## Scientific Purposes

---
## Workflow
### Proposed Pipeline

1. **Grid initialization**
    
    **Action:** Define the spatial and spectral boundaries of the volume of interest.
    
    **Dimensions:** Initialize a 3D array: [X, Y, Z].
    • `X, Y`: 2D spatial coordinates (projected physical distance, e.g., kiloparsecs or arcseconds).
    • `Z`: Spectral/Velocity space ($\Delta v$ in km/s).
    
2. **Ion fraction calculation (with `Trident` or `PyGad`)**
    
    **Action:** Determine the fractions of each ion per gas particle. 
    
    **The Math:** Interpolate these particle values against a pre-computed grid (usually from **Cloudy**).
    
3. **Emissivity calculation**
    
    **Action:** Determine how much light each gas particle actually emits, based on gas properties.
    
    **Input Data:** Extract `PartType0` (Gas) properties from the SIMBA snapshot:
    • Number Density ($n$)
    • Internal Energy $\rightarrow$ Temperature (T)
    • Metallicity ($Z$)
    
    **Processes:**
    
    1. Gravitational cooling → collisional ionization/excitation
    2. Photoionization (UV background, nearby young stars) → recombination
    
    **Output Data:** Derived emissivity per particle in the volume.
    
    **The Math:** $\epsilon = \epsilon(n, T, Z)$
    
4. **SPH kernel projection/deposition (likely with `yt`)**
    
    **Action:** Map the discrete 3D particles onto the 3D grid.
    
    **Spatial Splatting:**
    • Use the particle's smoothing length (h) and an SPH kernel function (e.g., cubic spline).
    • Distribute the particle's total emitted luminosity across the overlapping [X,Y] spatial pixels.
    
    **Spectral Splatting:**
    • Extract the particle's peculiar velocity vector.
    • Doppler-shift the emission to the correct velocity bin (Z-axis).
    • Apply thermal broadening based on the particle's temperature (Gaussian width).
    
5. **Emissivity to Surface Brightness**
    
    **Action:** Determine how many fluxes travel to the observer.
    
    **The Math:** Emissivity → intrinsic Luminosity → observed Flux, $F = \frac{L}{4\pi D_L^2}$
    
6. **Dust attenuation (Monte Carlo radiative transfer) (`SKIRT`)**
    
    **Action:** Account for the light lost as it travels out of the galaxy.
    
    **Method:** Integrate the dust density along the line of sight from the particle to the observer.
    
    **The Math:** Apply a dust extinction curve (e.g., Calzetti or Cardelli) to calculate the optical depth of the dust $\tau_{dust}$.
    
    **Correction:** Attenuate the emitted flux: $F_{obs} = F_{emit} \times e^{-\tau_{dust}}$.
    
    **Further concerns:** How do dust sizes and compositions affect the attenuation?
    
7. **Instrument convolution (mocking the telescopes)**
    
    **Action:** Degrade the perfect theoretical datacube to match realistic telescope limitations.
    
    **Spatial Blur (Seeing):** Convolve the [X,Y] spatial slices with the telescope's Point Spread Function (PSF).
    
    **Spectral Blur (Resolution):** Convolve the Z-velocity axis with the spectrograph's Line Spread Function (LSF).
    
    **Noise:** Add Gaussian or Poisson noise to mimic realistic Signal-to-Noise Ratios (SNR) based on expected E-ELT exposure times.
    
8. **Outputs**

```
main_controller():
	particle_ids = simba.select_halo_particle_ids()
	redshift = simba.get_redshift()
	
	emissivity = config["emissivity"]     # read from .config
	
	result = run_line_emission_pipeline()
		gas_fields = simba.load_physical_gas_fields()
		emission = calculate_particle_line_luminosity()
		result = make_xxv_cube()
```

---
## Extra Notes

## Running the current controller

Copy `config.example.json`, replace its paths, physical unit factors, center,
line label, and bin edges, then run from the project root:

```bash
python Pipeline/scr/pipeline.py Pipeline/your_config.json
```

The controller uses `input_source` to choose a gas adapter and
`emissivity.type` to choose an emissivity-table adapter. Currently, the
implemented choices are `simba` and `ps20`. Adding another source or table
requires an adapter with the same output fields; after that, a run selects it
in the JSON configuration.

Simulation inputs live in sibling modules such as `scr/simba.py`. A future
`scr/eagle.py` or `scr/illustris.py` can provide `select_particle_ids`,
`load_physical_gas_fields`, and `get_redshift`, then be registered in
`SOURCE_MODULES` in `scr/pipeline.py`. The older `scr/gas_sources.py` is
retained for comparison and is no longer used by the active controller.

`selection.type` can be `caesar_host_halo` as shown in the example. CAESAR
provides snapshot row indices; the SIMBA adapter translates those to
`ParticleIDs` before the main pipeline reads gas fields. To start from a saved
NumPy array of IDs instead, use:

```json
"selection": {
  "type": "particle_ids_file",
  "filename": "/path/to/particle_ids.npy"
}
```

The controller uses each selected ParticleID once. A repeated ID in the
selection is removed before calculating emissivity, so it cannot contribute
twice to the cube. The output records the number removed.

The result is a compressed `.npz` file containing `cube_erg_s` in
`(velocity, image_y, image_x)` order, its bin edges, and run metadata. The
cube is an intrinsic particle-luminosity sum with no spatial smoothing,
thermal broadening, dust attenuation, or instrument model yet.
