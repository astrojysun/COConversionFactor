# COConversionFactor

Python implementation of existing CO-to-H2 conversion factor prescriptions in the literature.

## Table of content

**[`CO_conversion_factor.alphaCO`](https://github.com/astrojysun/COConversionFactor/blob/main/CO_conversion_factor/alphaCO.py)** - This module includes a collection of conversion factor prescriptions
+ `predict_alphaCO10_S20`: a simple metallicity-dependent prescription (see [Sun et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...892..148S)).
+ `predict_alphaCO10_N12`: the [Narayanan et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.421.3127N) prescription.
+ `predict_alphaCO10_B13`: a refined version of the [Bolatto et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013ARA%26A..51..207B) prescription.
+ `predict_alphaCO10_B13_original`: the original [Bolatto et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013ARA%26A..51..207B) prescription.
+ `predict_alphaCO10_A16`: the [Amor&iacute;n et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016A&A...588A..23A) prescription.
+ `predict_alphaCO10_A17`: the [Accurso et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.4750A) prescription.
+ `predict_alphaCO_G20`: the [Gong et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...903..142G) prescription.
+ `predict_alphaCO21_T24`: the [Teng et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJ...961...42T) prescription.
+ `predict_alphaCO_SL24`: the [Schinnerer & Leroy (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240319843S) prescription.
+ (suggestions for other prescriptions are welcome)

**[`CO_conversion_factor.metallicity`](https://github.com/astrojysun/COConversionFactor/blob/main/CO_conversion_factor/metallicity.py)** - This module includes tools for predicting metallicity from scaling relations
+ `predict_logOH_SAMI19`: predict the gas phase abundance from mass-metallicity relations reported in [S&aacute;nchez et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.3042S).
+ `predict_logOH_CALIFA17`: predict the gas phase abundance from mass-metallicity relations reported in [S&aacute;nchez et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.2121S).
+ `extrapolate_logOH_radially`: extrapolate the abundance within a galaxy assuming a radial gradient (e.g., [S&aacute;nchez et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...563A..49S)).

## Installation

You can install the package directly from PyPI (https://pypi.org/project/CO-conversion-factor/) via

```
pip install --user CO_conversion_factor
```

## Dependencies

+ [`numpy`](https://numpy.org/)
+ [`scipy`](https://scipy.org/)
+ [`astropy`](https://www.astropy.org/)

## Credits

If you use tools from this repo in a publication, please make sure to cite the relevant source papers (see links above).

Please also consider acknowledgements to [`astropy`](https://github.com/astropy/astropy) and the other required packages.
