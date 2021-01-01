from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from astropy import units as u


def predict_logOH_SAMI19(
        Mstar, calibration='PP04', form='MZR', return_residual=False):
    """
    Predict 12+log(O/H) with the 'SAMI19' MZR (Sanchez+19).

    This function predicts the gas phase abundance 12+log(O/H)
    from the global stellar mass of a galaxy according to the
    mass-metallicity relations reported in Sanchez+19.
    Reference: Sanchez et al. (2019), MNRAS, 484, 3042

    Parameters
    ----------
    Mstar : number, `~numpy.ndarray`, `~astropy.units.Quantity` object
        Galaxy global stellar mass, in units of solar mass
    calibration : {'O3N2-M13', 'PP04', 'N2-M13', 'ONS', 'R23', ...}
        Metallicity calibration to adopt (see Table 1 in Sanchez+19).
        Default is 'PP04'.
    form : {'MZR', 'pMZR'}
        The MZR functional form to adopt (see Table 1 in Sanchez+19).
        Default is 'MZR'.
    return_residual : bool
        Whether to return the residual scatter around the MZR.
        Default is to not return.

    Returns
    -------
    logOH : number or `~numpy.ndarray`
        Predicted gas phase abundance, in units of 12+log(O/H)
    """

    calibrations = [
        'O3N2-M13', 'PP04', 'N2-M13', 'ONS', 'R23', 'pyqz', 't2',
        'M08', 'T04', 'EPM09', 'DOP16']
    if calibration not in calibrations:
        raise ValueError(
            "Available choices for `calibration` are: "
            "{}".format(calibrations))

    if hasattr(Mstar, 'unit'):
        x = np.log10(Mstar.to(u.Msun).value) - 8
    else:
        x = np.log10(Mstar) - 8

    # Mass-metallicity relation
    if form == 'MZR':  # MZR best fit
        params = np.zeros(
            3, dtype=[(calib, 'f') for calib in calibrations])
        params[0] = (  # a
            8.51, 8.73, 8.50, 8.51, 8.48, 9.01, 8.84,
            8.88, 8.84, 8.54, 8.94)
        params[1] = (  # b
            0.007, 0.010, 0.008, 0.011, 0.004, 0.017, 0.008,
            0.010, 0.007, 0.002, 0.020)
        params[2] = (  # sigma_MZR
            0.102, 0.147, 0.105, 0.138, 0.101, 0.211, 0.115,
            0.169, 0.146, 0.074, 0.288)
        a = params[calibration][0]
        b = params[calibration][1]
        c = 3.5
        residual = params[calibration][2]
        logOH = a + b*(x-c)*np.exp(-(x-c))

    elif form == 'pMZR':  # pMZR polynomial fit
        params = np.zeros(
            5, dtype=[(calib, 'f') for calib in calibrations])
        params[0] = (  # p0
            8.478, 8.707, 8.251, 8.250, 8.642, 8.647, 8.720,
            8.524, 8.691, 8.456, 8.666)
        params[1] = (  # p1
            -0.529, -0.797, -0.207, -0.428, -0.589, -0.718, -0.487,
            -0.148, -0.200, -0.097, -0.991)
        params[2] = (  # p2
            0.409, 0.610, 0.243, 0.427, 0.370, 0.682, 0.415,
            0.218, 0.164, 0.130, 0.738)
        params[3] = (  # p3
            -0.076, -0.113, -0.048, -0.086, -0.063, -0.133, -0.080,
            -0.040, -0.023, -0.032, -0.114)
        params[4] = (  # sigma_pMZR
            0.077, 0.112, 0.078, 0.101, 0.087, 0.143, 0.087,
            0.146, 0.123, 0.071, 0.207)
        p0 = params[calibration][0]
        p1 = params[calibration][1]
        p2 = params[calibration][2]
        p3 = params[calibration][3]
        residual = params[calibration][4]
        logOH = p0 + p1*x + p2*x**2 + p3*x**3

    else:
        raise ValueError("Invalid input value for `form`!")

    if return_residual:
        return logOH, residual
    else:
        return logOH


def predict_logOH_CALIFA17(
        Mstar, calibration='PP04', return_residual=False):
    """
    Predict 12+log(O/H) with the 'CALIFA17' MZR (Sanchez+17).

    This function predicts the gas phase abundance 12+log(O/H)
    from the global stellar mass of a galaxy according to the
    mass-metallicity relations reported in Sanchez+17.
    Reference: Sanchez et al. (2017), MNRAS, 469, 2121

    Parameters
    ----------
    Mstar : number, `~numpy.ndarray`, `~astropy.units.Quantity` object
        Galaxy global stellar mass, in units of solar mass
    calibration : {'O3N2-M13', 'PP04', 'N2-M13', 'ONS', 'R23', ...}
        Metallicity calibration to adopt (see Table 1 in Sanchez+19).
        Default is 'PP04'.
    return_residual : bool
        Whether to return the residual scatter around the MZR.
        Default is to not return.

    Returns
    -------
    logOH : number or `~numpy.ndarray`
        Predicted gas phase abundance, in units of 12+log(O/H)
    """

    calibrations = [
        'O3N2-M13', 'PP04', 'N2-M13', 'ONS', 'R23', 'pyqz', 't2',
        'M08', 'T04', 'EPM09', 'DOP16']
    if calibration not in calibrations:
        raise ValueError(
            "Available choices for `calibration` are: "
            "{}".format(calibrations))

    if hasattr(Mstar, 'unit'):
        x = np.log10(Mstar.to(u.Msun).value) - 8
    else:
        x = np.log10(Mstar) - 8

    # Mass-metallicity relation
    params = np.zeros(
        3, dtype=[(calib, 'f') for calib in calibrations])
    params[0] = (  # a
        8.53, 8.76, 8.53, 8.55, 8.54, 9.00, 8.85,
        8.72, 8.92, 8.59, 8.86)
    params[1] = (  # b
        0.003, 0.005, 0.004, 0.006, 0.003, 0.007, 0.007,
        0.004, 0.008, 0.001, 0.008)
    params[2] = (  # sigma_MZR
        0.060, 0.087, 0.060, 0.082, 0.065, 0.147, 0.064,
        0.087, 0.133, 0.060, 0.183)
    a = params[calibration][0]
    b = params[calibration][1]
    c = 3.5
    residual = params[calibration][2]
    logOH = a + b*(x-c)*np.exp(-(x-c))

    if return_residual:
        return logOH, residual
    else:
        return logOH


def extrapolate_logOH_radially(
        logOH_Re, gradient='CALIFA14', Rgal=None, Re=None):
    """
    Extrapolate 12+log(O/H) assuming a fixed radial gradient.

    This function extrapolates the gas phase abundance 12+log(O/H)
    from its value at 1 Re to the entire galaxy, according to a fixed
    radial gradient specified by the user.

    Parameters
    ----------
    logOH_Re : number, `~numpy.ndarray`
        Gas phase abundance at 1 Re, in units of 12+log(O/H)
    gradient : {'CALIFA14', float}
        Radial abundance gradient to adopt, in units of dex/Re.
        Default is 'CALIFA14', i.e., -0.10 dex/Re
        (Reference: Sanchez et al. 2014, A&A, 563, A49).
    Rgal : number, ndarray, Quantity object
        Galactocentric radii, in units of kilo-parsec
    Re : number, ndarray, Quantity object
        Galaxy effective radii, in units of kilo-parsec

    Returns
    -------
    logOH : number or `~numpy.ndarray`
        Predicted gas phase abundance, in units of 12+log(O/H)
    """

    if (Rgal is None) or (Re is None):
        return logOH_Re
    
    if hasattr(Rgal, 'unit') and hasattr(Re, 'unit'):
        Rgal_normalized = (Rgal / Re).to('').value
    elif hasattr(Rgal, 'unit') or hasattr(Re, 'unit'):
        raise ValueError(
            "`Rgal` and `Re` should both carry units "
            "or both be dimensionless")
    else:
        Rgal_normalized = np.asarray(Rgal) / np.asarray(Re)

    # metallicity gradient
    if gradient == 'CALIFA14':
        alpha_logOH = -0.10  # dex/Re
        logOH = (logOH_Re + alpha_logOH * (Rgal_normalized - 1))
    else:
        alpha_logOH = gradient  # dex/Re
        logOH = (logOH_Re + alpha_logOH * (Rgal_normalized - 1))

    return logOH



