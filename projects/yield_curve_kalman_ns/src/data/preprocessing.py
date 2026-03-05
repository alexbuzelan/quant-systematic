import pandas as pd

def prepare_yield_dataframe(
    df: pd.DataFrame,
    maturity_map: dict,
    convert_to_decimal: bool = False
) -> pd.DataFrame:
    """
    Prepare a yield dataframe for Nelson-Siegel / DNS estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw yield dataframe (index=dates, columns=tickers like 'DGS10').

    maturity_map : dict
        Mapping from column name -> maturity in years.

    convert_to_decimal : bool
        If True, convert percent yields (e.g. 4.25) to decimals (0.0425).

    Returns
    -------
    pd.DataFrame
        Clean dataframe with:
        - datetime index
        - numeric maturity columns (years)
        - sorted maturities
    """

    yields_df = df.copy()

    # Rename columns to maturities
    yields_df = yields_df.rename(columns=maturity_map)

    # Convert percent -> decimal
    if convert_to_decimal:
        yields_df = yields_df / 100.0

    # Ensure datetime index
    yields_df.index = pd.to_datetime(yields_df.index)

    # Keep only numeric maturity columns
    yields_df = yields_df[[c for c in yields_df.columns if isinstance(c, (int, float))]]

    # Sort maturities
    yields_df = yields_df.reindex(sorted(yields_df.columns), axis=1)

    # Drop rows where all yields are missing
    yields_df = yields_df.dropna(how="all")

    return yields_df

# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------

if __name__ == "__main__":

    # Example: mock raw yield data
    data = {
        "DATE": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "DGS1": [4.10, 4.12, 4.11],
        "DGS2": [4.00, 4.02, 4.01],
        "DGS10": [3.90, 3.92, 3.91],
    }

    raw_df = pd.DataFrame(data).set_index("DATE")

    # Define maturity mapping
    maturity_map = {
        "DGS1": 1.0,
        "DGS2": 2.0,
        "DGS10": 10.0,
    }

    # Prepare dataframe
    yields_df = prepare_yield_dataframe(
        raw_df,
        maturity_map,
        convert_to_decimal=True
    )

    print("Prepared yield dataframe:")
    print(yields_df.head())