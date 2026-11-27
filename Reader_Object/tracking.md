# Track selected gas across SIMBA snapshots

Run these cells from the project directory with your emission `catalogue` open.
Use complete, unsliced, single-file snapshots from the **same simulation run**.
The first version matches gas (`PartType0`) IDs, not halo membership or particle ancestry.

## 1. Register snapshots

```python
from Reader_Object import read_RawSnapshots

raw_snap = read_RawSnapshots(files=[
    '/path/to/snapshot_116.hdf5',
    '/path/to/snapshot_118.hdf5',
    '/path/to/snapshot_120.hdf5',
])
```

Opening reads metadata only. Paths retain your input order. Backward direction
is checked using `Header/Time`, not filenames. Basenames must be unique.
Physical fields use each snapshot's own scale factor and Hubble parameter,
with constants matching `config.example.yaml`. For different pipeline constants,
pass `unit_inputs=parameters['unit_factors']` when opening the collection
(the configuration constants, not the derived `physical_unit_factors`).

## 2. Track a selection

`mask` selects emission catalogue rows. `snapshot_row` contains rows in its
original raw snapshot. Set `current_snapshot` to that actual source file.

```python
mask = slice(0, 100)  # Or a Boolean mask matching the catalogue length
gas_rows = catalogue.get_fields('snapshot_row', selection=mask)
ids = catalogue.get_fields('particle_id', selection=mask)

tracks = raw_snap.track_backward(
    selected_rows=gas_rows,
    expected_particle_ids=ids,
    current_snapshot='snapshot_120.hdf5',
    target_snapshots=['snapshot_118.hdf5', 'snapshot_116.hdf5'],
    ptype='gas',
)
summary = tracks.describe()
```

Each target is matched against the original IDs independently. A missing match
at 118 does not prevent a match at 116. Arrays keep the original selection order.
Each selected gas ID must occur at most once per snapshot. If a selected ID
has multiple matches, the error gives the snapshot name and example IDs.
Duplicates outside the selection do not affect the result.
Matching metadata is a consistency check, not proof of a shared simulation run.

```python
rows_118 = tracks.rows['snapshot_118.hdf5']
found_118 = tracks.found['snapshot_118.hdf5']
```

**Missing rows are `-1`. Do not index snapshot fields with these arrays directly.**
`found=False` means absent from the gas ID array, not necessarily destroyed.
Type changes and particle ancestry need a separate, future tracking model.

## 3. Read aligned physical fields

```python
past_gas = tracks.get_fields(
    snapshot='snapshot_118.hdf5',
    fields=['T_K', 'mass_g', 'position_kpc', 'velocity_kms'],
    chunk_size=100000,
)

temperature = past_gas.values['T_K']  # Masked NumPy array, K
found = past_gas.found               # Explicit Boolean array
positions = past_gas.values['position_kpc']  # Shape (N_selected, 3)
```

Missing entries are masked, never replaced by physical zeros. Field reads
verify matched IDs again. Positions are absolute physical kpc in the simulation
box. Velocities are physical peculiar km/s in the simulation frame. These
are not centered on the earlier halo or corrected for its bulk velocity.

The first match reads each participating snapshot's entire gas ID column.
Sorted ID/row caches use about 16 bytes per gas particle per snapshot, plus
temporary sorting memory. Use a compute session for full SIMBA snapshots.
Physical fields are then read only for matched particles, in chunks.

```python
raw_snap.clear_cache()  # Free ID indexes; existing tracks still support field reads
```

Treat registered files as immutable. Reopen the collection if snapshots change.
