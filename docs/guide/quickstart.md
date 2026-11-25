# Quickstart

## Set up the environment

Run the following with Python 3.14, which the current [Octavius release](https://pypi.org/project/octavius/) requires. The pipeline runs directly from the checkout.

```bash
git clone --branch develop https://github.com/JiaHeak/SIMBA-gas-emission.git
cd SIMBA-gas-emission
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy h5py pyyaml octavius matplotlib
```

## Gather the inputs

1. A single-file SIMBA HDF5 snapshot with its original gas row order.
2. The [Octavius catalogue](https://octavius.readthedocs.io/en/latest/guide/catalogues.html) from that exact snapshot, with halo gas membership and the `r200m`, `minpot_pos`, and `com_vel_total` halo fields.
3. One HDF5 emissivity table containing your lines. Follow [Cloudy table preparation](cloudy.md) to convert a completed Cloudy run, or use a PS20 line-emissivity table.

## Run for one halo

```bash
cp config.example.yaml config.yaml
```

Edit the snapshot, catalogue, and table paths in `config.yaml`. Choose an Octavius `selection.target_index` and copy the exact line labels from the table. Set a new `output.filename`. The [configuration reference](configuration.md) explains these settings.

```bash
python scr/pipeline.py config.yaml
```

The pipeline prints the number of particles and lines saved and the path to the [particle emission catalogue](output.md). With the example output name, open that file and plot a map and spectrum with the [Reader example](../examples/analysis.md).

## Run on Slurm

For a large snapshot, save the following as `run_pipeline.slurm` and submit it from the repository root with `sbatch run_pipeline.slurm`. Replace the memory, time, and environment settings to match your job and cluster.

```bash
#!/bin/bash
#SBATCH --job-name=simba-emission
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=simba-emission-%j.log

set -eu
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python scr/pipeline.py config.yaml
```
