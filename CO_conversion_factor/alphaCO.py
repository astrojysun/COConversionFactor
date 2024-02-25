from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
from scipy.optimize import newton
from astropy import units as u, constants as const

# Galactic conversion factor value (Bolatto+13)
alphaCO10_Galactic = 4.35 * u.Unit('Msun s / (pc2 K km)')


def predict_alphaCO10_S20(Zprime=None):
    """
    Predict alphaCO10 with a simple power-law metallicity dependence.

    This is the default prescription used in Sun et al. (2020).
    It is similar to the metallicity-dependent part of the
    xCOLD-GASS prescription (Accurso+17).

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
        WCO10kpc=None, Sigmaelsekpc=None, Sigmatotkpc=None,
        T_ex=30., **kwargs):
    """
    Predict alphaCO10 with the (refined) Bolatto+13 prescription.

    This function implements a refined version of the prescription
    suggested by Bolatto+13 in both non-iterative and iterative modes.
    + In the non-iterative mode, it uses `Zprime` and `Sigmatotkpc` to
      calculate alphaCO10 directly.
    + In the iterative mode, it uses `Zprime`, `WCO10kpc`, and
      `Sigmaelsekpc` to iteratively solve for alphaCO10. This mode
      is usually more useful for observers.

    *Important*
    (1) The GMC surface density term in the original formula is
    INTENTIONALLY SUPPRESSED here to avoid non-sensical solutions.
    (2) The predicted alphaCO TRUNCATES at a lower boundary set by the
    optically thin limit.

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
    WCO10kpc : number or array-like or `~astropy.units.Quantity`
        Integrated CO(1-0) intensity on kpc-scale, in units of K*km/s.
    Sigmaelsekpc : number or array-like or `~astropy.units.Quantity`
        Mass surface density of HI gas + stars on kpc-scale,
        in units of Msun/pc^2.
    Sigmatotkpc : number or array-like or `~astropy.units.Quantity`
        Mass surface density of all gas + stars on kpc-scale,
        in units of Msun/pc^2.
    T_ex : number
        CO excitation temperature (in Kelvin) used to calculate the
        optically thin limit. Default value is 30.
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
        alphaCO10 = 2.9 * np.exp(0.4 / np.atleast_1d(Zprime))
        # starburst correction
        if hasattr(Sigmatotkpc, 'unit'):
            Sigtotkpc100 = Sigmatotkpc.to('100 Msun / pc^2').value
        else:
            Sigtotkpc100 = np.atleast_1d(Sigmatotkpc) / 100
        f_SB = Sigtotkpc100**-0.5
        f_SB[f_SB > 1] = 1
        alphaCO10 = alphaCO10 * f_SB
    else:
        if hasattr(WCO10kpc, 'unit'):
            WK = WCO10kpc.to('K km / s').value
        else:
            WK = WCO10kpc
        if hasattr(Sigmaelsekpc, 'unit'):
            SK = Sigmaelsekpc.to('Msun / pc2').value
        else:
            SK = Sigmaelsekpc
        Zp, WK, SK = np.broadcast_arrays(Zprime, WK, SK)
        alphaCO10 = []
        x0 = alphaCO10_Galactic.value
        for Zp_, WK_, SK_ in zip(Zp.ravel(), WK.ravel(), SK.ravel()):
            if not ((Zp_ > 0) and (WK_ > 0) and (SK_ > 0)):
                alphaCO10 += [np.nan]
                continue
            def func(x):
                return predict_alphaCO10_B13(
                    iterative=False, Zprime=Zp_,
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

    # clip at optically thin limit
    alphaCO10_OTL = (
        0.34 * (T_ex / 30) * np.exp(5.53/T_ex - 5.53/30))
    alphaCO10[alphaCO10 < alphaCO10_OTL] = alphaCO10_OTL

    if alphaCO10.size == 1:
        alphaCO10 = alphaCO10.item()
    return alphaCO10 * u.Unit('Msun s / (pc2 K km)')


def predict_alphaCO10_B13_original(
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
        J='1-0', r_beam=None, r_thresh=100*u.pc,
        Zprime=None, R_21=None, T_peak=None, W_CO=None):
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
      3) `r_beam` >= `r_thresh` and no `R_21` (line ratio) is provided.

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
    r_thresh : number or `~astropy.units.Quantity`
        Resolution threshold (in units of parsec), above which the
        'Z only' method is preferred over 'Z & T_peak' or 'Z & W_CO'.
        Default is 100 pc.
    Zprime : number or array-like
        Metallicity normalized to the solar value
    R_21 : number or array-like
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
    if hasattr(r_thresh, 'unit'):
        r_th = r_thresh.to('pc').value
    else:
        r_th = r_thresh
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
    elif r_ >= r_th and R_21 is None:
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
                np.min([r_, r_th])**0.081)
        else:
            XCO = (
                1.5e20 *
                (np.atleast_1d(R_21)/0.6)**-1.69 *
                np.atleast_1d(Zprime)**-0.50 *
                np.min([r_, r_th])**0.063)
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


def predict_alphaCO21_T24(
        vdisp_150pc=None, vdisp_150pc_lolim=3., vdisp_150pc_uplim=30.,
        Zprime=None):
    """
    Predict alphaCO with the Teng+24 prescription.

    This is the recommended prescription in Teng+24 (Eq. 2 therein).

    Reference: Teng et al. (2024), ApJ, 961, 42

    Parameters
    ----------
    vdisp_150pc : number or array-like or `~astropy.units.Quantity`
        Average molecular gas velocity dispersion (in km/s unit)
        measured on 150 pc scales.
    vdisp_150pc_lolim : number or `~astropy.units.Quantity`
        Lower limit of molecular gas velocity dispersion (in km/s unit),
        at which the prescription is clipped.
    vdisp_150pc_uplim : number or `~astropy.units.Quantity`
        Upper limit of molecular gas velocity dispersion (in km/s unit),
        at which the prescription is clipped.
    Zprime : number or array-like
        Metallicity normalized to the solar value. If given, it will
        only be used to decide if a warning message is triggered.

    Returns
    -------
    alphaCO21 : `~astropy.units.Quantity` object
        Predicted CO(2-1)-to-H2 conversion factor, carrying a unit of
        Msun/pc^2/(K*km/s).
    """
    if vdisp_150pc is None:
        raise ValueError(
            "`vdisp_150pc` is necessary for predicting alphaCO21")

    if hasattr(vdisp_150pc, 'unit'):
        vdisp = vdisp_150pc.to('km s-1').value
    else:
        vdisp = vdisp_150pc
    if hasattr(vdisp_150pc_lolim, 'unit'):
        vdisp_lolim = vdisp_150pc_lolim.to('km s-1').value
    else:
        vdisp_lolim = vdisp_150pc_lolim
    if hasattr(vdisp_150pc_uplim, 'unit'):
        vdisp_uplim = vdisp_150pc_uplim.to('km s-1').value
    else:
        vdisp_uplim = vdisp_150pc_uplim

    if Zprime is not None:
        if Zprime < 0.5:
            print(
                "WARNING: prescription likely not appropriate at "
                "< 0.5 solar metallicity")

    alphaCO21 = np.minimum(np.maximum(
        vdisp, vdisp_lolim), vdisp_uplim)**-0.81 * 10**1.05 / 0.65

    return alphaCO21 * u.Unit('Msun s / (pc2 K km)')


def predict_alphaCO_SL24(
        J='1-0', Zprime=None, Sigma_star=None, Sigma_sfr=None,
        metal_pl=-1.5, Zprime_uplim=2.0, Zprime_lolim=0.2,
        stellar_pl=-0.25, Sigma_star_thresh=1e2, Sigma_star_uplim=1e3,
        sfr_pl=0.125, Sigma_sfr_norm=1.8e-2, rco_norm=None,
        rco_lolim=None, rco_uplim=None, return_all_terms=False):
    """
    Predict alphaCO with the Schinnerer & Leroy (2024) prescription.

    This function implements a set of prescriptions suggested by
    Schinnerer & Leroy (2024). It predicts conversion factors for
    the CO(1-0), CO(2-1), or CO(3-2) lines based on metallicity,
    stellar surface density, and SFR surface density on kpc scales.

    Reference: Schinnere & Leroy (2024), ARA&A

    Parameters
    ----------
    J : {'1-0', '2-1', '3-2'}
        CO transition in question
    Zprime : number or array-like
        Metallicity normalized to the solar value
    Sigma_star : number or array-like or `~astropy.units.Quantity`
        Stellar mass surface density (in Msun/pc^2 unit).
    Sigma_sfr : number or array-like or `~astropy.units.Quantity`
        Star formation rate surface density (in Msun/yr/kpc^2 unit).


    Other parameters
    ----------------
    metal_pl : number (default: -1.5)
        Power-law index on metallicity for the CO-dark term.
    Zprime_uplim : number (default: 2.0)
        Upper limit of Zprime, at which the CO-dark term is clipped.
    Zprime_lolim : number (default: 0.2)
        Lower limit of Zprime, at which the CO-dark term is clipped.
    stellar_pl : number (default: -0.25)
        Power-law index on stellar (mass) surface density for the
        starburst term.
    Sigma_star_thresh : number or Quantity (default: 1e2)
        Stellar surface density threshold (in Msun/pc^2 unit),
        above which the starburst term is turned on.
    Sigma_star_uplim : number or Quantity (default: 1e3)
        Upper limit of stellar surface density (in Msun/pc^2 unit),
        at which the starburst term is clipped.
    sfr_pl : number (default: 0.125)
        Power-law index on SFR surface density for CO line ratio.
    Sigma_sfr_norm : number or Quantity (default: 1.8e-2)
        Normalization of the SFR surface density (in Msun/yr/kpc^2).
    rco_norm : number
        Normalization of the CO line ratio (ignored for J=1-0).
        Default is 0.65 for J=2-1 and 0.325 for J=3-2.
    rco_lolim : number
        Lower limit of the CO line ratio (ignored for J=1-0).
        Default is 0.35 for J=2-1 and 0.175 for J=3-2.
    rco_uplim : number
        Upper limit of the CO line ratio (ignored for J=1-0).
        Default is 1.0 for J=2-1 and 0.5 for J=3-2.
    return_all_terms : bool (default: False)
        Whether to return all intermediate terms (CO-dark, starburst,
        line ratio) together with the predicted conversion factor.

    Returns
    -------
    alphaCO : `~astropy.units.Quantity` object
        Predicted CO-to-H2 conversion factor, carrying a unit of
        Msun/pc^2/(K*km/s).
    additional_returns
        If `return_all_terms` is set to True, then the intermediate
        terms (CO-dark, starburst, line ratio) are also returned.
    """
    if J not in ('1-0', '2-1', '3-2'):
        raise ValueError(
            "Prescriptions are only available for"
            "the J=1-0, 2-1, and 3-2 transitions")

    if Zprime is None or Sigma_star is None:
        raise ValueError(
            "Both `Zprime` and `Sigma_star` are needed for"
            "predicting alphaCO")
    if Sigma_sfr is None and J != '1-0':
        raise ValueError(
            "`Sigma_sfr` is needed when J != 1-0")

    if J == '2-1':
        if rco_norm is None:
            rco_norm = 0.65
        if rco_lolim is None:
            rco_lolim = 0.35
        if rco_uplim is None:
            rco_uplim = 1.0
    elif J == '3-2':
        if rco_norm is None:
            rco_norm = 0.325
        if rco_lolim is None:
            rco_lolim = 0.175
        if rco_uplim is None:
            rco_uplim = 0.5

    if hasattr(Sigma_star, 'unit'):
        Sigstar = Sigma_star.to('Msun pc-2').value
    else:
        Sigstar = Sigma_star
    if hasattr(Sigma_star_thresh, 'unit'):
        Sigstar_thresh = Sigma_star_thresh.to('Msun pc-2').value
    else:
        Sigstar_thresh = Sigma_star_thresh
    if hasattr(Sigma_star_uplim, 'unit'):
        Sigstar_uplim = Sigma_star_uplim.to('Msun pc-2').value
    else:
        Sigstar_uplim = Sigma_star_uplim
    if hasattr(Sigma_sfr, 'unit'):
        Sigsfr = Sigma_sfr.to('Msun yr-1 kpc-2').value
    else:
        Sigsfr = Sigma_sfr
    if hasattr(Sigma_sfr_norm, 'unit'):
        Sigsfr_norm = Sigma_sfr_norm.to('Msun yr-1 kpc-2').value
    else:
        Sigsfr_norm = Sigma_sfr_norm

    # CO-dark term
    f_term = np.minimum(
        np.maximum(Zprime, Zprime_lolim), Zprime_uplim
    ) ** metal_pl

    # starburst term
    g_term = np.minimum(
        np.maximum(Sigstar/Sigstar_thresh, 1),
        Sigstar_uplim/Sigstar_thresh
    ) ** stellar_pl

    # CO line ratio
    if J == '1-0':
        rco = 1.0
    else:
        rco = rco_norm * (Sigsfr / Sigsfr_norm) ** sfr_pl
        rco = np.minimum(np.maximum(rco, rco_lolim), rco_uplim)

    alphaCO = alphaCO10_Galactic * f_term * g_term / rco

    if return_all_terms:
        return alphaCO, f_term, g_term, rco
    else:
        return alphaCO
