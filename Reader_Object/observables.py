"""Conversions from intrinsic, line-integrated emission to observed quantities."""
from math import isfinite, pi
from numbers import Real

KPC_TO_CM = 3.0856775814913673e21
ARCSEC_PER_RADIAN = 180 * 3600 / pi
SURFACE_BRIGHTNESS_UNIT = 'erg s^-1 cm^-2 arcsec^-2'


def surface_brightness_factor(redshift):
    """Convert isotropic line luminosity per physical kpc² to observed brightness.

    The ratio of angular-diameter to luminosity distance squared gives
    the Tolman factor (1+z)^-4. The result applies to energy integrated
    across the line and includes the conversion from steradians to arcsec².

    Returns: factor (float): observed surface brightness per erg s^-1 kpc^-2.
    """
    if (isinstance(redshift, bool) or not isinstance(redshift, Real)
            or not isfinite(redshift) or redshift <= -1):
        raise ValueError('redshift must be a finite number greater than -1.')
    factor = (1 + float(redshift))**-4 / (4 * pi * KPC_TO_CM**2 * ARCSEC_PER_RADIAN**2)
    if factor == 0:
        raise ValueError('Surface-brightness conversion underflowed at this redshift.')
    return factor
