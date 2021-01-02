from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.optimize import newton
from astropy import units as u, constants as const

# Galactic conversion factor value (Bolatto+13)
alphaCO10_Galactic = 4.35 * u.Unit('Msun s / (pc2 K km)')


def predict_alphaCO10_PHANGS(Zprime=None):
    """
    Predict alphaCO10 with a simple metallicity-dependent prescription.

    This is the default prescription used in several PHANGS papers
    (e.g., Sun+20). It is similar to the metallicity-dependent part of
    the xCOLD-GASS prescription (Accurso+17).

    Reference: Accurso et al. 2017, MNRAS, 470, 4750;
               Sun et al. 2020, ApJ, 892, 148

    Parameters
    ----------
    Zprime : number or array-like
        Metallicity normalized to the solar value

    Returns
    -------
    alphaCO10 : `~astropy.units.Quantity` object
        Predicted CO-to-H2 conversion factor, carrying a unit of
        Msun/pc^2/(K*km/s).
    """
    alphaCO10 = alphaCO10_Galactic * np.atleast_1d(Zprime)**-1.6

    if alphaCO10.size == 1:
        alphaCO10 = alphaCO10.item()
    return alphaCO10


def predict_alphaCO10_N12(Zprime=None, WCO10GMC=None):
    """
    Predict alphaCO10 with the Narayanan+12 prescription.
    
    This corresponds to Equation 11 in Narayanan+12.
    Also note that an additional multiplicative factor of 1.35 is
    included here to correct for Helium contribution (it is not
    included in the original formula in this paper).

    Reference: Narayanan et al. (2012), MNRAS, 421, 3127
    
    Parameters
    ----------
    Zprime : number or array-like
        Metallicity normalized to the solar value
    WCO10GMC : number or array-like or `~astropy.units.Quantity`
        Luminosity-weighted CO(1-0) intensity over all GMCs,
        in units of K*km/s.

    Returns
    -------
    alphaCO10 : `~astropy.units.Quantity` object
        Predicted CO-to-H2 conversion factor, carrying a unit of
        Msun/pc^2/(K*km/s).
    """
    if hasattr(WCO10GMC, 'unit'):
        WCO = WCO10GMC.to('K km / s').value
    else:
        WCO = WCO10GMC

    alphaCO10 = 10.7 * np.atleast_1d(WCO)**-0.32
    alphaCO10[alphaCO10 > 6.3] = 6.3
    alphaCO10 = alphaCO10 / np.atleast_1d(Zprime)**0.65
    alphaCO10 *= 1.35  # include Helium contribution

    if alphaCO10.size == 1:
        alphaCO10 = alphaCO10.item()
    return alphaCO10 * u.Unit('Msun s / (pc2 K km)')


def predict_alphaCO10_B13(
        iterative=True, suppress_error=False, Zprime=None,
        WCO10GMC=None, WCO10kpc=None, Sigmaelsekpc=None,
        SigmaGMC=None, Sigmatotkpc=None, **kwargs):
    """
    Predict alphaCO10 with the Bolatto+13 prescription.
    
    This function implements a prescription suggested by Bolatto+13
    (Equation 31 therein) in both non-iterative and iterative modes.
    + In the non-iterative mode, it uses `Zprime`, `SigmaGMC`, and
      `Sigmatotkpc` to calculate alphaCO10 directly.
    + In the iterative mode, it uses `Zprime`, `WCO10GMC`, `WCO10kpc`,
      and `Sigmaelsekpc` to iteratively solve for alphaCO10. This mode
      is usually more useful for observers.

    Reference: Bolatto et al. (2013), ARA&A, 51, 207
    
    Parameters
    ----------
    iterative : bool
        Whether to solve for alphaCO10 iteratively (default: True).
        See description above in the docstring.
    suppress_error : bool
        Whether to suppress error if iteration fail to converge.
        Default is to not suppress.
    Zprime : number or array-like
        Metallicity normalized to the solar value
    WCO10GMC : number or array-like or `~astropy.units.Quantity`
        Integrated CO(1-0) intensity of GMCs, in units of K*km/s.
    WCO10kpc : number or array-like or `~astropy.units.Quantity`
        Integrated CO(1-0) intensity on kpc-scale, in units of K*km/s.
    Sigmaelsekpc : number or array-like or `~astropy.units.Quantity`
        Mass surface density of HI gas + stars on kpc-scale,
        in units of Msun/pc^2.
    SigmaGMC : number or array-like or `~astropy.units.Quantity`
        Mass surface density of molecular gas in GMCs,
        in units of Msun/pc^2.
    Sigmatotkpc : number or array-like or `~astropy.units.Quantity`
        Mass surface density of all gas + stars on kpc-scale,
        in units of Msun/pc^2.
    **kwargs
        Keywords to be passed to `~scipy.optimize.newton`.
        Useful only when using the iterative prescriptions.

    Returns
    -------
    alphaCO10 : `~astropy.units.Quantity` object
        Predicted CO-to-H2 conversion factor, carrying a unit of
        Msun/pc^2/(K*km/s).
    """
    if not iterative:
        # CO faint correction
        if hasattr(SigmaGMC, 'unit'):
            SigGMC100 = SigmaGMC.to('100 Msun / pc^2').value
        else:
            SigGMC100 = np.atleast_1d(SigmaGMC) / 100
        alphaCO10 = 2.9 * np.exp(
            0.4 / np.atleast_1d(Zprime) / SigGMC100)
        # starburst correction
        if hasattr(Sigmatotkpc, 'unit'):
            Sigtotkpc100 = Sigmatotkpc.to('100 Msun / pc^2').value
        else:
            Sigtotkpc100 = np.atleast_1d(Sigmatotkpc) / 100
        f_SB = Sigtotkpc100**-0.5
        f_SB[f_SB > 1] = 1
        alphaCO10 = alphaCO10 * f_SB
    else:
        if hasattr(WCO10GMC, 'unit'):
            WG = WCO10GMC.to('K km / s').value
        else:
            WG = WCO10GMC
        if hasattr(WCO10kpc, 'unit'):
            WK = WCO10kpc.to('K km / s').value
        else:
            WK = WCO10kpc
        if hasattr(Sigmaelsekpc, 'unit'):
            SK = Sigmaelsekpc.to('Msun / pc2').value
        else:
            SK = Sigmaelsekpc
        Zp, WG, WK, SK = np.broadcast_arrays(Zprime, WG, WK, SK)
        alphaCO10 = []
        x0 = alphaCO10_Galactic.value
        for Zp_, WG_, WK_, SK_ in zip(
                Zp.ravel(), WG.ravel(), WK.ravel(), SK.ravel()):
            if not ((Zp_ > 0) and (WG_ > 0) and
                    (WK_ > 0) and (SK_ > 0)):
                alphaCO10 += [np.nan]
                continue
            def func(x):
                return predict_alphaCO10_B13(
                    iterative=False,
                    Zprime=Zp_,
                    SigmaGMC=WG_*x,
                    Sigmatotkpc=(WK_*x+SK_)).value - x
            if suppress_error:
                try:
                    alphaCO10 += [newton(func, x0, **kwargs)]
                except RuntimeError as e:
                    print(e)
                    alphaCO10 += [np.nan]
            else:
                alphaCO10 += [newton(func, x0, **kwargs)]
        alphaCO10 = np.array(alphaCO10).reshape(Zp.shape)
        
    if alphaCO10.size == 1:
        alphaCO10 = alphaCO10.item()
    return alphaCO10 * u.Unit('Msun s / (pc2 K km)')


def predict_alphaCO10_A16(logOH=None):
    """
    Predict alphaCO10 with the Amorin+16 prescription.

    This is the best-fit relation in Amorin+16 (Figure 11 therein).

    Reference: Amorin et al. (2016), A&A, 588, A23

    Parameters
    ----------
    logOH : number or array-like
        Gas phase abundance, in units of 12+log(O/H).
        Note that Amorin+16 adopted an N2-based calibration
        (Perez-Montero & Contini 2009).
        
    Returns
    -------
    alphaCO10 : `~astropy.units.Quantity` object
        Predicted CO-to-H2 conversion factor, carrying a unit of
        Msun/pc^2/(K*km/s).
    """
    alphaCO10 = 10**(0.68 - 1.45 * (np.atleast_1d(logOH) - 8.7))

    if alphaCO10.size == 1:
        alphaCO10 = alphaCO10.item()
    return alphaCO10 * u.Unit('Msun s / (pc2 K km)')


def predict_alphaCO10_A17(logOH=None, Mstar=None, SFR=None, z=None):
    """
    Predict alphaCO10 with the Accurso+17 prescription.

    This prescription includes a 1st-order dependence on metallicity
    and a 2nd-order dependence on Delta(MS). If any of the keywords
    `Mstar`, `SFR`, or `z` is not specified, the secondary
    dependence will be turned off.

    Reference: Accurso et al. (2017), MNRAS, 470, 4750

    Parameters
    ----------
    logOH : number or array-like
        Gas phase abundance, in units of 12+log(O/H).
        Note that Accurso+17 adopted an O3N2-based calibration
        (Pettini & Pagel 2004).
    Mstar : number or array-like or `~astropy.units.Quantity`
        Galaxy global stellar mass, in units of Msun
    SFR : number or array-like or `~astropy.units.Quantity`
        Gaalxy global star formation rate, in units of Msun/yr
    z : number or array-like
        Galaxy redshift
        
    Returns
    -------
    alphaCO10 : `~astropy.units.Quantity` object
        Predicted CO-to-H2 conversion factor, carrying a unit of
        Msun/pc^2/(K*km/s).
    """
    alphaCO10 = 10**(14.752 - 1.623 * np.atleast_1d(logOH))

    if Mstar is not None and SFR is not None and z is not None:
        if hasattr(Mstar, 'unit'):
            logMstar = np.log10(Mstar.to('Msun').value)
        else:
            logMstar = np.log10(Mstar)
        if hasattr(SFR, 'unit'):
            logSFR = np.log10(SFR.to('Msun/yr').value)
        else:
            logSFR = np.log10(SFR)
        log_sSFR = logSFR - logMstar + 9
        log_sSFR_MS = (
            -1.12 + 1.14*z - 0.19*z**2 -
            (0.3 + 0.13*z) * (logMstar - 10.5))
        log_Delta_MS = log_sSFR - log_sSFR_MS
        alphaCO10 = alphaCO10 * 10**(0.062 * log_Delta_MS)
        
    if alphaCO10.size == 1:
        alphaCO10 = alphaCO10.item()
    return alphaCO10 * u.Unit('Msun s / (pc2 K km)')


def predict_alphaCO_G20(
        J='1-0', r_beam=None, Zprime=None,
        R_21=None, T_peak=None, W_CO=None):
    """
    Predict alphaCO with the Gong+20 prescription.

    This function implements a set of prescriptions suggested by
    Gong+20 (Table 3 therein). It provides conversion factors
    for both CO(1-0) and CO(2-1) lines based on metallicity,
    observable CO line properties, and resolution of the CO data.
    
    Depending on the availability of input variables, one of the
    following four methods will be used to predict alphaCO:

    + 'Z only' method -- this is adopted if:
      1) only `Zprime` (metallicity) is available; or
      2) `r_beam` (CO data resolution) is unspecified; or
      3) `r_beam` >= 100pc and no `R_21` (CO line ratio) is provided.

    + 'Z & R_21' method -- this is adopted if both `Zprime` and `R_21`
      are available.

    + 'Z & T_peak' method -- this is adopted if both `Zprime` and
      `T_peak` (CO line peak temperature) are available, but `R_21` is
      not available.

    + 'Z & W_CO' method -- this is adopted if both `Zprime` and `W_CO`
      (CO line integrated intensity) are available, but neither `R_21`
      nor `T_peak` is available.

    Reference: Gong et al. (2020), ApJ, 903, 142

    Parameters
    ----------
    J : {'1-0', '2-1'}
        CO transition in question
    r_beam : number or `~astropy.units.Quantity`
        Resolution (beam size) of the CO observation,
        in units of parsec.
    Zprime : number or array-like
        Metallicity normalized to the solar value
    R_21 : number or array-like or `~astropy.units.Quantity`
        CO(2-1) to CO(1-0) line ratio at the specified resolution
    T_peak : number or array-like or `~astropy.units.Quantity`
        CO line peak temperature at the specified resolution,
        in units of Kelvin.
    W_CO : number or array-like or `~astropy.units.Quantity`
        CO line integrated intensity at the specified resolution,
        in units of K*km/s.

    Returns
    -------
    alphaCO : `~astropy.units.Quantity` object
        Predicted CO-to-H2 conversion factor, carrying a unit of
        Msun/pc^2/(K*km/s).
    """
    if J not in ('1-0', '2-1'):
        raise ValueError(
            "Prescriptions are only available for"
            "the J=1-0 and J=2-1 transitions")

    if Zprime is None:
        raise ValueError(
            "`Zprime` is required for predicting alphaCO")

    if hasattr(r_beam, 'unit'):
        r_ = r_beam.to('pc').value
    else:
        r_ = r_beam
    if hasattr(T_peak, 'unit'):
        T_ = T_peak.to('K').value
    else:
        T_ = T_peak
    if hasattr(W_CO, 'unit'):
        W_ = W_CO.to('K km s-1').value
    else:
        W_ = W_CO

    if r_ is None:
        method = 'Z only'
    elif r_ >= 100 and R_21 is None:
        method = 'Z only'
    else:
        if R_21 is not None:
            method = 'Z & R_21'
        elif T_ is not None:
            method = 'Z & T_peak'
        elif W_ is not None:
            method = 'Z & W_CO'
        else:
            method = 'Z only'

    if method == 'Z only':
        if J == '1-0':
            XCO = 1.4e20 * np.atleast_1d(Zprime)**-0.80
        else:
            XCO = 2.0e20 * np.atleast_1d(Zprime)**-0.50
    elif method == 'Z & R_21':
        if J == '1-0':
            XCO = (
                0.93e20 *
                (np.atleast_1d(R_21)/0.6)**-0.87 *
                np.atleast_1d(Zprime)**-0.80 *
                np.min([r_, 100])**0.081)
        else:
            XCO = (
                1.5e20 *
                (np.atleast_1d(R_21)/0.6)**-1.69 *
                np.atleast_1d(Zprime)**-0.50 *
                np.min([r_, 100])**0.063)
    elif method == 'Z & T_peak':
        if J == '1-0':
            XCO = (
                1.8e20 *
                np.atleast_1d(T_)**(-0.64+0.24*np.log10(r_)) *
                np.atleast_1d(Zprime)**-0.80 * r_**-0.083)
        else:
            XCO = (
                2.7e20 *
                np.atleast_1d(T_)**(-1.07+0.37*np.log10(r_)) *
                np.atleast_1d(Zprime)**-0.50 * r_**-0.13)
    else:
        if J == '1-0':
            XCO = (
                6.1e20 *
                np.atleast_1d(W_)**(-0.54+0.19*np.log10(r_)) *
                np.atleast_1d(Zprime)**-0.80 * r_**-0.25)
        else:
            XCO = (
                21.1e20 *
                np.atleast_1d(T_)**(-0.97+0.34*np.log10(r_)) *
                np.atleast_1d(Zprime)**-0.50 * r_**-0.41)

    if XCO.size == 1:
        XCO = XCO.item()
    alphaCO = (
        XCO * u.Unit('cm-2 K-1 km-1 s') *
        const.u * 1.00794 * 2).to('Msun s / (pc2 K km)')
    alphaCO *= 1.35  # include Helium contribution
    return alphaCO
    
            
        
