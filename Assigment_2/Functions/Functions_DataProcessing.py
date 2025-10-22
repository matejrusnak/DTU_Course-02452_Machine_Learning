import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_dataset(path:str) -> pd.DataFrame:
    return pd.read_csv(path)

def column_drop(df: pd.DataFrame, *column_name:  str) -> pd.DataFrame:
    return df.drop(columns = list(column_name))

def replace_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include = 'number').columns
    for col in numeric_cols:
        mean = df_copy[col].mean()
        df_copy[col] = df_copy[col].fillna(mean)
    return df_copy

def one_hot_encoder(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    dummies = pd.get_dummies(df[columns], dtype = float)

    return df.drop(columns = columns).join(dummies)

def power(df: pd.DataFrame, cols: list[str], exponent: int) -> pd.DataFrame:
    transformed_df = df.copy()
    transformed_df[cols] = transformed_df[cols] ** exponent
    return transformed_df

def log_transform(df: pd.DataFrame, cols: list[str], base: float = np.e) -> pd.DataFrame:
    transformed_df = df.copy()
    if base == 2:
        transformed_df[cols] = np.log2(transformed_df[cols])
    elif base == 10:
        transformed_df[cols] = np.log10(transformed_df[cols])
    else:
        transformed_df[cols] = np.log(transformed_df[cols])
    return transformed_df

def standartize(df) -> pd.DataFrame:
    scaler = StandardScaler()
    return scaler.fit(df)