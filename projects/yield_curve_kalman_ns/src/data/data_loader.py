import pandas as pd
import re
from pathlib import Path

def read_fred_yield_xlsx(path: Path, sheet="Daily") -> pd.DataFrame:
    """
    Reads one FRED yield Excel file (e.g. DGS10.xlsx) from the 'Daily' sheet,
    returns a 2-col DataFrame: observation_date + <maturity_code>.
    """
    df = pd.read_excel(path, sheet_name=sheet)

    # Standardize date column name
    if "observation_date" not in df.columns:
        # sometimes it's "DATE" etc; try to find it
        date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
        if date_col is None:
            raise ValueError(f"No date-like column found in {path.name}. Columns: {df.columns.tolist()}")
        df = df.rename(columns={date_col: "observation_date"})

    # Identify the yield column (usually the ticker, e.g. DGS10)
    # Prefer a column matching the filename stem; else take the first non-date column.
    ticker = path.stem  # "DGS10"
    if ticker in df.columns:
        ycol = ticker
    else:
        non_date_cols = [c for c in df.columns if c != "observation_date"]
        if not non_date_cols:
            raise ValueError(f"No yield column found in {path.name}.")
        ycol = non_date_cols[0]
        df = df.rename(columns={ycol: ticker})
        ycol = ticker

    # Parse dates
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")

    # Clean yield: handle '.' or ',' decimals and common missing-value markers
    s = df[ycol].astype(str).str.strip()
    s = s.replace({"nan": pd.NA, ".": pd.NA, "": pd.NA, "NA": pd.NA, "N/A": pd.NA})
    # Convert "3,22" -> "3.22"
    s = s.str.replace(",", ".", regex=False)
    # Keep only numeric-looking strings (optional but helps with weird footnotes)
    s = s.where(s.str.match(r"^-?\d+(\.\d+)?$", na=False), pd.NA)

    df[ycol] = pd.to_numeric(s, errors="coerce")

    # Keep only the two columns we need
    df = df[["observation_date", ycol]].dropna(subset=["observation_date"])
    return df


def load_yield_curve_panel(folder: str, pattern="DGS*.xlsx", sheet="Daily") -> pd.DataFrame:
    """
    Loads all matching FRED yield Excel files and constructs a yield curve panel.

    Each file should contain a maturity (e.g. DGS1, DGS2, DGS10) with a column
    'observation_date' and the yield series. Files are merged on the date index
    so that only dates where all maturities are available are kept.

    Parameters
    ----------
    folder : str
        Path to the folder containing the FRED Excel files.
    pattern : str, optional
        File pattern used to identify the files (default: "DGS*.xlsx").
    sheet : str, optional
        Name of the Excel sheet containing the data (default: "Daily").

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by observation_date with maturities as columns
        (e.g. DGS1, DGS2, ..., DGS30).

    Example
    -------
    >>> from src.data.fred_loader import load_yield_curve_panel
    >>> yc = load_yield_curve_panel("data/raw/fred_yields")
    >>> yc.head()

    This will produce a DataFrame like:

                     DGS1   DGS2   DGS5   DGS10   DGS30
    observation_date
    2010-01-04       0.45   0.93   2.65    3.85    4.65
    2010-01-05       0.47   0.95   2.70    3.88    4.68
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {folder}")

    dfs = []
    for f in files:
        dfs.append(read_fred_yield_xlsx(f, sheet=sheet))

    # Inner-join on dates => keeps only dates common to ALL maturities
    panel = dfs[0]
    for df in dfs[1:]:
        panel = panel.merge(df, on="observation_date", how="inner")

    # Ensure full cross-section each day (no missing yields)
    maturity_cols = [c for c in panel.columns if c != "observation_date"]
    panel = panel.dropna(subset=maturity_cols)

    panel = panel.sort_values("observation_date").set_index("observation_date")

    # Optional: sort columns by maturity number if you want (DGS1, DGS2, ..., DGS30)
    def maturity_key(col):
        m = re.search(r"(\d+)$", col)
        return int(m.group(1)) if m else 10**9

    panel = panel.reindex(sorted(maturity_cols, key=maturity_key), axis=1)
    return panel

if __name__ == "__main__":
    yc = load_yield_curve_panel("data/raw/fred_yields")
    print(yc.head())
    