"""
Data loading and preprocessing utilities.
Provides functions to load synthetic or real CSV data, aggregate counts by state/year/demographic,
apply Rule-of-11 suppression, and compute rates used for visualization and summarization.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load dataset from CSV if provided, otherwise look for packaged synthetic CSV.
    The returned DataFrame uses a canonical schema expected by other modules.
    """
    if path and Path(path).exists():
        df = pd.read_csv(path)
    else:
        default = Path(__file__).parent / 'data' / 'synthetic_health.csv'
        if default.exists():
            df = pd.read_csv(default)
        else:
            raise FileNotFoundError(f"No dataset found at {path or default}. Run generate_synthetic.py to create it.")

    # Normalize columns if needed
    expected = ['patient_id','state','year','age_group','sex','race_ethnicity','income_group','heart_disease','diabetes','cancer']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")

    # Ensure types
    df['year'] = df['year'].astype(int)
    df['state'] = df['state'].astype(str)
    for col in ['heart_disease','diabetes','cancer']:
        df[col] = df[col].astype(int)

    return df


def aggregate_by_state(df: pd.DataFrame, disease: str, groupby: list = ['state','year'], denominator_col: Optional[str]=None) -> pd.DataFrame:
    """Aggregate counts and compute rates per state/year or other grouping.
    Returns DataFrame with columns: groupby..., cases, population, rate
    If denominator_col is None we approximate population by counting records.
    """
    denom = denominator_col or 'patient_id'
    agg = df.groupby(groupby).agg(cases=(disease, 'sum'), population=(denom, 'count')).reset_index()
    agg['rate'] = agg['cases'] / agg['population']
    return agg


def apply_rule_of_11(df: pd.DataFrame, case_col: str = 'cases', pop_col: str = 'population') -> pd.DataFrame:
    """Suppress small counts to comply with Rule of 11.
    Any cell where cases < 11 or population < 11 will have `suppressed=True` and rate set to NaN.
    """
    df = df.copy()

    # If empty, return a minimal-sane DataFrame shape
    if df.empty:
        # Ensure expected columns exist so callers can rely on them
        df = df.copy()
        if 'rate' not in df.columns:
            df['rate'] = np.nan
        df['suppressed'] = pd.Series(dtype=bool)
        if case_col not in df.columns:
            df[case_col] = pd.Series(dtype=float)
        if pop_col not in df.columns:
            df[pop_col] = pd.Series(dtype=float)
        return df

    # Reset index to avoid assignment errors on exotic indices
    df = df.reset_index(drop=True)

    # Ensure case/pop columns exist (fallback to zeros if missing)
    for c in (case_col, pop_col):
        if c not in df.columns:
            df[c] = 0

    # Compute suppression mask safely
    mask = (df[case_col] < 11) | (df[pop_col] < 11)
    # Fill NA in mask with False
    mask = mask.fillna(False)
    df['suppressed'] = mask

    # Use the mask to assign NaN where suppressed
    if 'rate' not in df.columns:
        df['rate'] = np.nan

    df.loc[mask, 'rate'] = np.nan
    # Optionally mask cases/pop for display
    df.loc[mask, case_col] = np.nan
    return df


def filter_dataset(df: pd.DataFrame, disease: Optional[str]=None, year: Optional[int]=None, demographics: Dict[str, Any]=None) -> pd.DataFrame:
    """Filter dataset by disease/year/demographics. `demographics` is a dict like {'income_group':'Low'}.
    disease arg is not used for row filtering (diseases are columns) but kept for API symmetry.
    """
    out = df.copy()
    if year is not None:
        out = out[out['year'] == int(year)]

    # Mapping from possible frontend/display demographic keys to the canonical dataframe columns
    display_to_col = {
        'age': 'age_group',
        'agegroup': 'age_group',
        'age group': 'age_group',
        'race': 'race_ethnicity',
        'race_ethnicity': 'race_ethnicity',
        'race ethnicity': 'race_ethnicity',
        'income': 'income_group',
        'income level': 'income_group',
        'income_group': 'income_group',
    }

    if demographics:
        for k, v in demographics.items():
            if v is None:
                continue

            # normalize key (allow frontend display labels like 'Race' or 'Income Level')
            key_norm = str(k).strip().lower()
            col = None
            if key_norm in display_to_col:
                col = display_to_col[key_norm]
            elif k in out.columns:
                col = k
            else:
                # try case-insensitive column match
                matches = [c for c in out.columns if c.lower() == key_norm]
                if matches:
                    col = matches[0]

            if col is None:
                allowed = list(display_to_col.keys()) + list(out.columns)
                raise ValueError(f"Unknown demographic filter key '{k}'. Allowed keys (examples): {allowed}")

            out = out[out[col] == v]

    return out
