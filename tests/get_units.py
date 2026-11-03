"""This is a script to inspect the Units used in the Snapshot."""

import h5py

with h5py.File("your_snapshot.hdf5", "r") as f:
    print(dict(f["Header"].attrs))
    for field in ("Coordinates", "Velocities", "Masses", "Density", "InternalEnergy"):
        dataset = f[f"PartType0/{field}"]
        print(field, dataset.shape, dict(dataset.attrs))