import pandas as pd
import numpy as np

def drop_high_na_columns(df, threshold=0.99):       
    """
    This function identifies columns whose share of missing values exceeds
    the specified threshold and removes them from the DataFrame.
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
        threshold (float): Proportion of allowed missing values. Columns with
            missing-value ratios above this threshold are dropped.
    
    Returns:
        tuple:
            pandas.DataFrame: DataFrame with high-NA columns removed.
            list[str]: List of column names that were dropped.
    
    Raises:
        ValueError: If `threshold` is not between 0 and 1.
    """
    # Drop high NAN-Value columns
    high_nan = df.isna().mean()
    cols_to_drop = high_nan[high_nan > threshold].index
    return df.drop(columns=cols_to_drop), list(cols_to_drop)


def fill_missing_values(df):
    """
    Fill missing values using domain-specific rules.

    This function applies custom imputation strategies:
    - children: missing values are replaced with 0
    - country: missing values are replaced with "Donno"
    - agent/company: missing values are replaced with 0
    
    Args:
        df (pandas.DataFrame): Input DataFrame with missing values.
    
    Returns:
        pandas.DataFrame: DataFrame with filled missing 
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
    Ensure correct datatypes after filling missing values.

    This function converts numeric identifier columns to integer type:
    - agent
    - company
    - children
    
    Args:
        df (pandas.DataFrame): Input DataFrame.
    
    Returns:
        pandas.DataFrame: DataFrame with corrected datatypes.
    
    Raises:
        ValueError: If required columns are missing or contain non-numeric values.
    """
    # Convert numeric IDs to integers
    df["agent"] = df["agent"].astype(int)
    df["company"] = df["company"].astype(int)
    df["children"] = df["children"].astype(int)

    return df


def remove_outliers(df):
    """
    Remove extreme outliers based on domain logic.

    This function removes rows with unrealistic values:
    - lead_time: rows above the 99th percentile are removed
    - adr: rows above the 99th percentile are removed
    
    Args:
        df (pandas.DataFrame): Input dataset.
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame with outliers removed.
    """
    # Lead time: remove top 1% outliers because nobody will book 2 years in advance
    q99_lead = df["lead_time"].quantile(0.99)
    df = df[df["lead_time"] <= q99_lead]

    # ADR: remove top 1% (very high nightly price) => 5400$ is unrealisticly high 
    q99_adr = df["adr"].quantile(0.99)
    df = df[df["adr"] <= q99_adr]

    return df


def clean_dataset(df):
    """
    Run the full cleaning pipeline in the correct order.

    This function executes all preprocessing steps:
    1. Drop columns with excessive missing values
    2. Fill missing values using domain rules
    3. Correct datatypes
    4. Remove extreme outliers
    5. Reset index
    
    Args:
        df (pandas.DataFrame): Raw input dataset.
    
    Returns:
        tuple:
            pandas.DataFrame: Fully cleaned dataset.
            list[str]: Names of columns dropped due to high NA share.
    """
    # Run pipeline
    df_clean, dropped_cols = drop_high_na_columns(df)
    df_clean = fill_missing_values(df_clean)
    df_clean = fix_dtypes(df_clean)
    df_clean = remove_outliers(df_clean)
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean, dropped_cols
