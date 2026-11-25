# SIMBA CGM line-emission pipeline

Calculate intrinsic line emission from the circumgalactic gas of a SIMBA halo. The pipeline uses a SIMBA snapshot, its matching Octavius halo catalogue, and a Cloudy or PS20 emissivity table. It writes a reusable HDF5 file with the selected particles and their emission in every requested line.

Start with the [quickstart](guide/quickstart.md) to generate a catalogue, or go straight to the [analysis examples](examples/analysis.md) if you already have one. Maps, spectra, profiles, and line ratios can be prepared from the same file.

[GitHub repository](https://github.com/JiaHeak/SIMBA-gas-emission/tree/develop)

```{toctree}
:caption: Getting started
:hidden:

guide/quickstart
```

```{toctree}
:caption: User guide
:hidden:

guide/configuration
guide/cloudy
guide/output
guide/limitations
```

```{toctree}
:caption: Examples
:hidden:

examples/analysis
Reader notebook walkthrough <examples/reader>
examples/tracking
```
