import pandas as pd
import numpy as np


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path: str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the contents of the CSV file.
    """
    return pd.read_csv(path)


def column_drop(df: pd.DataFrame, *, column_name:  list[str]) -> pd.DataFrame:
    """
    Drop one or more specified columns from a DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    column_name: list[str]
        One or more column names to drop.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified columns removed.
    """
    return df.drop(columns=list(column_name))


def replace_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in a numeric columns with the column mean.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where NaN values in numeric columns are replaced
        with the mean of their respective columns. Non-numeric columns
        are left unchanged.
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include = 'number').columns
    for col in numeric_cols:
        mean = df_copy[col].mean()
        df_copy[col] = df_copy[col].fillna(mean)
    return df_copy


def one_hot_encoder(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Perform one-hot encoding on specified categorical columns.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    columns: (list[str])
        List of column names to apply one-hot encoding to.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where the specified categorical columns are replaced
        with their one-hot encoded representations. Non-specified columns
        are left unchanged.
    """
    dummies = pd.get_dummies(df[columns], dtype=float)
    return df.drop(columns=columns).join(dummies)


def power(df: pd.DataFrame, *, cols: list[str], exponent: int) -> pd.DataFrame:
    """
    Raise values in specified DataFrame columns to a given power.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : list[str]
        One or more column names to transform.
    exponent : int
        Exponent to which the column values will be raised.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where the specified columns are raised
        to the given power. Other columns remain unchanged.
    """
    transformed_df = df.copy()
    transformed_df[cols] = transformed_df[cols] ** exponent
    return transformed_df


def log_transform(df: pd.DataFrame, *, cols: list[str], base: float = np.e) -> pd.DataFrame:
    """
    Apply a logarithmic transformation to specified DataFrame columns.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    cols: list[str]
        One or more column names to transform.
    base: float
        Base of the logarithm. Supported values are 2, 10, or e (natural log).

    Returns
    -------
    pd.DataFrame
        A new DataFrame where the specified columns are transformed
        using the logarithm with the given base. Other columns remain unchanged.
    """
    transformed_df = df.copy()
    if base == 2:
        transformed_df[cols] = np.log2(transformed_df[cols])
    elif base == 10:
        transformed_df[cols] = np.log10(transformed_df[cols])
    else:
        transformed_df[cols] = np.log(transformed_df[cols])
    return transformed_df
