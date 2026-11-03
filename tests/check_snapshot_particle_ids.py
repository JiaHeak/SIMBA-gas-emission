"""Count repeated gas ParticleIDs in a SIMBA HDF5 snapshot."""

import argparse

import h5py
import numpy as np


def check_particle_ids(snapshot_filename: str) -> dict:
    """Read all gas IDs and count values that occur more than once.

    Returns: result (dict): particle count, duplicate counts, and example IDs.
    """
    with h5py.File(snapshot_filename, 'r') as snapshot:
        dataset = snapshot['PartType0/ParticleIDs']
        print(f'Dataset: {dataset.name}', flush=True)
        print(f'Shape: {dataset.shape}; dtype: {dataset.dtype}', flush=True)
        particle_ids = dataset[...]

    unique_ids, counts = np.unique(particle_ids, return_counts=True)
    repeated = counts > 1
    repeated_ids = unique_ids[repeated]
    repeated_counts = counts[repeated]

    return {
        'particle_count': int(particle_ids.size),
        'unique_id_count': int(unique_ids.size),
        'repeated_id_count': int(repeated_ids.size),
        'extra_rows_with_repeated_ids': int(np.sum(repeated_counts - 1)),
        'examples': list(zip(repeated_ids[:10].tolist(), repeated_counts[:10].tolist())),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot_filename')
    args = parser.parse_args()

    result = check_particle_ids(args.snapshot_filename)
    for name, value in result.items():
        print(f'{name}: {value}', flush=True)
