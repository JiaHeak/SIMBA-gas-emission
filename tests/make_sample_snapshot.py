"""Create a tiny, explicitly synthetic HDF5 snapshot for reader tests.

This file mimics the group/dataset layout used by a GIZMO-style snapshot. Its
values are invented test data, not simulation output and not scientific data.
Run from the project root with:

    python Pipeline/tests/make_sample_snapshot.py
"""
from pathlib import Path

import h5py
import numpy as np


OUTPUT = Path(__file__).resolve().parent / "data" / "sample_snapshot.hdf5"


def make_sample_snapshot(output: Path = OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    # Eight particles are enough to test full reads, selected rows, field
    # alignment, vector fields and integer particle IDs.
    n = 8
    coords = np.arange(n * 3, dtype=np.float32).reshape(n, 3) * 0.25
    velocities = np.linspace(-70, 70, n * 3, dtype=np.float32).reshape(n, 3)
    mass = np.linspace(0.5, 1.2, n, dtype=np.float32)
    density = np.logspace(-5, -2, n, dtype=np.float32)
    internal_energy = np.linspace(100, 800, n, dtype=np.float32)
    electron_abundance = np.linspace(0.1, 1.0, n, dtype=np.float32)
    sfr = np.zeros(n, dtype=np.float32)

    # Example 11-column layout only: total metals, helium, C, N, O, Ne, Mg,
    # Si, S, Ca, Fe. Values are synthetic and are not a calibrated abundance.
    metallicity = np.zeros((n, 11), dtype=np.float32)
    metallicity[:, 0] = np.linspace(0.001, 0.02, n)
    metallicity[:, 1] = 0.24
    metallicity[:, 2:] = metallicity[:, [0]] * np.array(
        [0.18, 0.05, 0.42, 0.10, 0.05, 0.06, 0.03, 0.01, 0.10],
        dtype=np.float32,
    )

    with h5py.File(output, "w") as f:
        f.attrs["FileDescription"] = "SYNTHETIC TEST FIXTURE; NOT SIMBA DATA"
        header = f.create_group("Header")
        header.attrs["Redshift"] = 0.0
        header.attrs["Time"] = 1.0
        header.attrs["HubbleParam"] = 0.7
        header.attrs["BoxSize"] = 10.0
        header.attrs["NumPart_Total"] = np.array([n, 0, 0, 0, 0, 0], dtype=np.uint32)
        header.attrs["UnitConvention"] = "arbitrary test values; do not apply SIMBA conversions"

        gas = f.create_group("PartType0")
        datasets = {
            "Coordinates": coords,
            "Velocities": velocities,
            "Masses": mass,
            "Density": density,
            "InternalEnergy": internal_energy,
            "ElectronAbundance": electron_abundance,
            "Metallicity": metallicity,
            "ParticleIDs": np.arange(1001, 1001 + n, dtype=np.uint64),
            "StarFormationRate": sfr,
        }
        for name, values in datasets.items():
            gas.create_dataset(name, data=values)

    return output


if __name__ == "__main__":
    print(make_sample_snapshot())
