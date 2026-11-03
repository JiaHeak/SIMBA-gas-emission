"""Run from any folder: python read_particle_catalogue.py /path/to/particle_emission.hdf5"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Reader_Object import read_EmissionCatalogue

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('filename')
parser.add_argument('--line', help='Exact line label; default is the first saved line')
parser.add_argument('--width-kpc', type=float, default=200.)
args = parser.parse_args()

catalogue = read_EmissionCatalogue(args.filename)
print(catalogue.describe())
line = args.line or catalogue.line_labels[0]
# Defaults come from the YAML reference frame. center='halo' selects the
# original halo centre even if YAML specified a different default centre.
image = catalogue.make_map(line, los_axis='z', half_width_kpc=args.width_kpc, bins=256)
spectrum = catalogue.make_spectrum(line, np.linspace(-300, 300, 101),
                                   los_axis='z', aperture_radius_kpc=args.width_kpc)
catalogue.plot(image)
catalogue.plot(spectrum)
print('Map included/excluded luminosity:', image['included_luminosity_erg_s'], image['excluded_luminosity_erg_s'])
print('Spectrum included/excluded luminosity:', spectrum['included_luminosity_erg_s'], spectrum['excluded_luminosity_erg_s'])
plt.show()
