# SIMBA CGM line-emission pipeline

**One SIMBA halo · Multiple emission lines · One reusable particle catalogue**

```mermaid
flowchart TD
    A["SIMBA snapshot + matching Octavius catalogue"] --> B["Select halo CGM gas and convert units"]
    B --> C["Interpolate emissivity and calculate particle luminosity"]
    Q["Cloudy input, grid and emissivity files"] --> T["Convert to one HDF5 emissivity table"]
    T --> C
    C --> D["HDF5 emission catalogue (Particle-based)"]
    D --> |Reader_Object| E["Maps · Spectra · Profiles · Line ratios"] -.-> F
    D --> F["Selected particle IDs"]
    S["Original SIMBA snapshots"] --> G["Read physical gas fields and track earlier gas"]
    F --> G

    classDef input fill:#DBEAFE,stroke:#2563EB,color:#172554;
    classDef process fill:#DCFCE7,stroke:#15803D,color:#14532D;
    classDef data fill:#FEF3C7,stroke:#B45309,color:#78350F;
    classDef result fill:#EDE9FE,stroke:#7C3AED,color:#3B0764;

    class A,Q,S input;
    class B,C,G,T process;
    class D,F data;
    class E result;

        subgraph legend [Legend]
        direction LR
        L1["Input"]:::input
        L2["Our Pipeline"]:::process
        L3["Data product"]:::data
        L4["Analysis output"]:::result
    end
```

## Documentation

- [Quickstart](docs/guide/quickstart.md) to run the pipeline for one halo.
- [Configuration reference](docs/guide/configuration.md) and [example YAML](config.example.yaml).
- [Cloudy table preparation](docs/guide/cloudy.md) and [HDF5 output format](docs/guide/output.md).
- [Analysis examples](docs/examples/analysis.md), [Reader notebook walkthrough](Reader_Object/manual.md), and [gas tracking](Reader_Object/tracking.md).
- [Current limitations](docs/guide/limitations.md) for gas selection, emissivity models, and analysis.

## Build the documentation locally

Run from the repository root with Python 3.12.

```bash
python3.12 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. [.readthedocs.yaml](.readthedocs.yaml) configures the same Sphinx build for Read the Docs Community.
