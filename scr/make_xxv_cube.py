# This is a script to make position-position-velocity (XXV) cubes from the simulation data.

import numpy as np

def make_xxv_cube(gas_fields: dict, particle_luminosity_erg_s, x_edges_kpc, y_edges_kpc, velocity_edges_kms, *, 
                  los_axis: str = 'z', systemic_velocity_kms: float = 0.0) -> dict:
    """Bin particle luminosities into a simple position-position-velocity cube.

    ``gas_fields`` uses the keys from ``convert_gas_fields`` and
    ``particle_luminosity_erg_s`` can be passed from
    ``calculate_ps20_particle_line_luminosity``. The cube is returned in array
    order ``(velocity, y, x)``. Its values are summed particle luminosities
    per voxel (erg s^-1); no smoothing or spectral broadening is applied.

    The three Cartesian viewing directions use:
      - ``los_axis='z'``: image x-y, velocity v_z
      - ``los_axis='y'``: image x-z, velocity v_y
      - ``los_axis='x'``: image y-z, velocity v_x

    Spatial edges are in kpc and velocity edges are in km s^-1. Particles
    outside the supplied bin edges do not contribute to the cube.
    """
    image_axes = {
        'x': ('y_kpc', 'z_kpc', 'v_x_kms', 'y', 'z'),
        'y': ('x_kpc', 'z_kpc', 'v_y_kms', 'x', 'z'),
        'z': ('x_kpc', 'y_kpc', 'v_z_kms', 'x', 'y'),
    }
    image_x_key, image_y_key, velocity_key, image_x_name, image_y_name = image_axes[los_axis]

    image_x = gas_fields[image_x_key]
    image_y = gas_fields[image_y_key]
    line_of_sight_velocity = gas_fields[velocity_key] - systemic_velocity_kms
    particle_luminosity_erg_s = np.asarray(particle_luminosity_erg_s)

    # histogramdd bins in (x, y, velocity) order; transpose to the common
    # cube storage order (velocity, y, x).
    binned, _ = np.histogramdd(
        np.column_stack((image_x, image_y, line_of_sight_velocity)),
        bins=(x_edges_kpc, y_edges_kpc, velocity_edges_kms),
        weights=particle_luminosity_erg_s,
    )
    cube = binned.transpose(2, 1, 0)

    return {
        'cube_erg_s': cube,
        'image_x_edges_kpc': np.asarray(x_edges_kpc),
        'image_y_edges_kpc': np.asarray(y_edges_kpc),
        'velocity_edges_kms': np.asarray(velocity_edges_kms),
        'image_x_axis': image_x_name,
        'image_y_axis': image_y_name,
        'los_axis': los_axis,
    }