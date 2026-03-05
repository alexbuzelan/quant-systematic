import pandas as pd
import numpy as np

def ns_loadings(tau_years: np.ndarray, lam: float):
    """
    Compute Nelson–Siegel factor loadings.

    Parameters
    ----------
    tau_years : array-like
        Maturities in years.
    lam : float
        Decay parameter (lambda).

    Returns
    -------
    L1, L2 : np.ndarray
        Nelson–Siegel factor loadings.
    """
    tau = np.asarray(tau_years, dtype=float)
    x = lam * tau

    eps = 1e-12  # numerical tolerance for very small values
    L1 = np.where(np.abs(x) < eps, 1.0, (1.0 - np.exp(-x)) / x)
    L2 = L1 - np.exp(-x)

    return L1, L2


def fit_ns_betas_for_date(tau_years, y, lam, w=None):
    """
    Estimate Nelson–Siegel betas (beta0, beta1, beta2) for a single date.

    The Nelson–Siegel yield curve model is:

        y(tau) = beta0 + beta1 * L1(tau) + beta2 * L2(tau)

    where L1 and L2 are the Nelson–Siegel factor loadings determined by
    the decay parameter lambda.

    Parameters
    ----------
    tau_years : array-like
        Maturities in years (e.g. [1, 2, 3, 5, 10, 20, 30]).

    y : array-like
        Observed yields for the corresponding maturities at a single date.

    lam : float
        Nelson–Siegel decay parameter.

    w : array-like, optional
        Optional weights for each maturity. Useful if you want to give
        higher importance to some maturities (e.g. liquid benchmarks).

    Returns
    -------
    np.ndarray
        Array containing estimated betas:
        [beta0, beta1, beta2]

    Example
    -------
    tau_maturities = np.array([1, 2, 3, 5, 10, 20, 30])

    yields_today = np.array([
        0.030, 0.032, 0.033, 0.035, 0.038, 0.041, 0.042
    ])

    lam = 0.0609

    betas = fit_ns_betas_for_date(
        tau_maturities,
        yields_today,
        lam
    )

    Output:

        betas = [beta0, beta1, beta2]
        e.g. [0.041, -0.020, 0.015]
    """

    # Convert inputs to numpy arrays
    tau_years = np.asarray(tau_years, float)
    y = np.asarray(y, float)

    # Compute Nelson–Siegel loadings for the maturities
    L1, L2 = ns_loadings(tau_years, lam)

    # Build regression design matrix:
    # X = [1, L1(tau), L2(tau)]
    X = np.column_stack([np.ones_like(tau_years), L1, L2])

    # Ordinary Least Squares estimation
    if w is None:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Weighted Least Squares estimation
    else:
        w = np.asarray(w, float)
        ws = np.sqrt(w)

        # Apply weights to both X and y
        beta = np.linalg.lstsq(X * ws[:, None], y * ws, rcond=None)[0]

    return beta


    
def extract_dns_factors(yields_df: pd.DataFrame, lam: float, weights=None) -> pd.DataFrame:
    """
    Estimate Nelson–Siegel factors (beta0, beta1, beta2) for each date from a panel of observed yields.

    Parameters
    ----------
    yields_df : pd.DataFrame
        DataFrame containing observed yields with:
        - index : dates
        - columns : maturities (years)
        - values : yields

        Example:

            date        1     2     3     5     10    20    30
            ---------------------------------------------------
            2020-01-01  0.03  0.032 0.033 0.035 0.038 0.041 0.042
            2020-01-02  0.031 0.033 0.034 0.036 0.039 0.042 0.043

    lam : float
        Nelson–Siegel decay parameter.

    weights : None, array-like, or dict, optional
        Optional maturity weights used in estimation.

        - None → equal weights
        - array-like → one weight per maturity
        - dict → mapping maturity → weight

    Returns
    -------
    pd.DataFrame
        Time series of Nelson–Siegel factors:

            date        beta0   beta1   beta2
            ---------------------------------
            2020-01-01  ...
            2020-01-02  ...

    Example
    -------
    lam = 0.0609

    factors = extract_dns_factors(
        yields_df,
        lam
    )

    Output:

        date        beta0   beta1   beta2
        ----------------------------------
        2020-01-01  0.041   -0.020  0.015
        2020-01-02  0.042   -0.019  0.014
    """

    # Extract maturities from the DataFrame columns
    tau = np.array([float(c) for c in yields_df.columns], float)

    # Convert weights if provided
    if isinstance(weights, dict):
        w = np.array([weights[float(c)] for c in yields_df.columns], float)

    elif weights is None:
        w = None

    else:
        w = np.asarray(weights, float)

    betas = []

    # Estimate factors for each date
    for dt, row in yields_df.iterrows():

        y = row.values.astype(float)

        # Handle missing yields
        mask = np.isfinite(y)

        # Need at least 3 maturities to estimate the 3 parameters
        if mask.sum() < 3:
            betas.append([np.nan, np.nan, np.nan])
            continue

        beta = fit_ns_betas_for_date(
            tau[mask],
            y[mask],
            lam,
            None if w is None else w[mask]
        )

        betas.append(beta)

    # Build factor DataFrame
    factors = pd.DataFrame(
        betas,
        index=yields_df.index,
        columns=["beta0", "beta1", "beta2"]
    )

    return factors.dropna()


    
def reconstruct_yields_from_factor_df(factors_df: pd.DataFrame, tau_years, lam: float) -> pd.DataFrame:
    """
    Reconstruct Nelson–Siegel fitted yields from a time series of factor estimates.


    Parameters
    ----------
    factors_df : pd.DataFrame
        DataFrame containing Nelson–Siegel factors with:
        - index : dates
        - columns : ['beta0', 'beta1', 'beta2']

    tau_years : array-like
        Maturities (in years) at which the yield curve should be reconstructed.

    lam : float
        Nelson–Siegel decay parameter.


    Returns
    -------
    pd.DataFrame
        DataFrame of fitted yields:
        - index   : dates (same as factors_df)
        - columns : maturities (tau_years)
        - values  : reconstructed yields


    Example
    -------
    predicted_factors should be a DataFrame like:

        date        beta0   beta1   beta2
        ----------------------------------
        2020-01-01  0.030   -0.015  0.020
        2020-01-02  0.031   -0.014  0.019
        2020-01-03  0.032   -0.013  0.018

    Usage:

        lam = 0.0609
        tau_maturities = np.array([1, 2, 3, 5, 10, 20, 30])
        y_model = reconstruct_yields_from_factor_df(predicted_factors, tau_maturities, lam=lam)

    Output (y_model) will look like:

        date        1      2      3      5      10     20     30
        --------------------------------------------------------
        2020-01-01  ...    ...    ...    ...    ...    ...    ...
        2020-01-02  ...    ...    ...    ...    ...    ...    ...
        2020-01-03  ...    ...    ...    ...    ...    ...    ...
    """
    
    tau_years = np.asarray(tau_years, float)
    L1, L2 = ns_loadings(tau_years, lam)

    # Shape: (T,1) + (T,1)*(N,) etc -> broadcasting to (T,N)
    b0 = factors_df["beta0"].to_numpy()[:, None]
    b1 = factors_df["beta1"].to_numpy()[:, None]
    b2 = factors_df["beta2"].to_numpy()[:, None]

    y_hat = b0 + b1 * L1[None, :] + b2 * L2[None, :]

    return pd.DataFrame(y_hat, index=factors_df.index, columns=tau_years)


def ns_yield_curve_from_betas(tau_years, beta0, beta1, beta2, lam):
    """
    Compute a Nelson–Siegel yield curve for a given set of factor values.

    Parameters
    ----------
    tau_years : array-like
        Maturities in years at which the yield curve will be evaluated
        (e.g. [1, 2, 3, 5, 10, 20, 30]).

    beta0 : float
        Level factor of the Nelson–Siegel model (long-term yield component).

    beta1 : float
        Slope factor (controls short-end vs long-end slope).

    beta2 : float
        Curvature factor (controls the hump in medium maturities).

    lam : float
        Nelson–Siegel decay parameter (determines where the curvature peaks).

    Returns
    -------
    pd.Series
        Series of model-implied yields with:
        - index : maturities (tau_years)
        - values : Nelson–Siegel fitted yields

    Example
    -------
    tau_maturities = np.array([1, 2, 3, 5, 10, 20, 30])

    beta0 = 0.04
    beta1 = -0.02
    beta2 = 0.015
    lam = 0.0609

    y_curve = ns_yield_curve_from_betas(
        tau_maturities,
        beta0,
        beta1,
        beta2,
        lam
    )

    Output (y_curve) will look like:

        maturity
        1      0.030
        2      0.032
        3      0.034
        5      0.036
        10     0.039
        20     0.041
        30     0.042
    """

    # Ensure maturities are a numpy array
    tau_years = np.asarray(tau_years, dtype=float)

    # Compute Nelson–Siegel factor loadings for each maturity
    L1, L2 = ns_loadings(tau_years, lam)

    # Nelson–Siegel yield formula
    # y(tau) = beta0 + beta1 * L1(tau) + beta2 * L2(tau)
    y = beta0 + beta1 * L1 + beta2 * L2

    # Return as pandas Series indexed by maturity
    return pd.Series(y, index=tau_years)