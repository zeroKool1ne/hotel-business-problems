import pandas as pd
import numpy as np

def drop_high_na_columns(df, threshold=0.99):
    """
    drop columns with nan_values greater than the threshold.
    """
    high_nan = df.isna().mean()
    cols_to_drop = high_nan[high_nan > threshold].index
    return df.drop(columns=cols_to_drop), list(cols_to_drop)


def fill_missing_values(df):
    """
    Handling of missing values according to isna().
    """
    

    # children: replace 4 missing values with 0
    df["children"] = df["children"].fillna(0)

    # country: replace missing with 'Donno'
    df["country"] = df["country"].fillna("Donno")

    # agent, company: replace missing IDs with 0
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)

    return df


def fix_dtypes(df):
    """
    makes sure we have proper datatypes after filling
    """
    

    # Convert numeric IDs to integers
    df["agent"] = df["agent"].astype(int)
    df["company"] = df["company"].astype(int)
    df["children"] = df["children"].astype(int)

    return df


def remove_outliers(df):
    """
    Removes unrealistic extreme values based on domain logic.
    """
    df = df.copy()

    # Lead time: remove top 1% outliers because nobody will book 2 years in advance
    q99_lead = df["lead_time"].quantile(0.99)
    df = df[df["lead_time"] <= q99_lead]

    # ADR: remove top 1% (very high nightly price) => 5400$ is unrealisticly high 
    q99_adr = df["adr"].quantile(0.99)
    df = df[df["adr"] <= q99_adr]

    return df


def clean_dataset(df):
    """
    this is the main pipeline which will run all cleaning steps in the right order.
    """
    df_clean, dropped_cols = drop_high_na_columns(df)
    df_clean = fill_missing_values(df_clean)
    df_clean = fix_dtypes(df_clean)
    df_clean = remove_outliers(df_clean)
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean, dropped_cols
