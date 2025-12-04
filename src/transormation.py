import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class HotelTransformer:
    """
    Transformation pipeline as a class.

    - Adds engineered features
    - One-hot encodes categoricals
    - Optionally scales numeric columns
    """

    def __init__(self, cat_cols=None, num_cols=None, do_scaling: bool = True):
        self.cat_cols = cat_cols or []
        self.num_cols = num_cols or []
        self.do_scaling = do_scaling

        self.encoder: OneHotEncoder | None = None
        self.scaler: StandardScaler | None = None

    # ---------- internal helpers ----------

    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create simple meaningful features for modeling.
        """
        df = df.copy()
        df["total_stay"] = (
            df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        )
        df["total_guests"] = (
            df["adults"] + df["children"].fillna(0) + df["babies"]
        )
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply already-fitted OneHotEncoder to df.
        """
        if not self.cat_cols:
            return df

        if self.encoder is None:
            raise RuntimeError("Encoder is not fitted. Call `fit` or `fit_transform` first.")

        encoded = self.encoder.transform(df[self.cat_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(self.cat_cols),
            index=df.index,
        )

        df = df.drop(columns=self.cat_cols)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def _scale_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply already-fitted StandardScaler to numeric columns.
        """
        if not self.do_scaling or not self.num_cols:
            return df

        if self.scaler is None:
            raise RuntimeError("Scaler is not fitted. Call `fit` or `fit_transform` first.")

        df = df.copy()
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df

    # ---------- public API (sklearn-like) ----------

    def fit(self, df: pd.DataFrame):
        """
        Fit encoder and scaler on the given dataframe.
        Use this on the TRAIN set.
        """
        df_tmp = self.add_engineered_features(df)

        if self.cat_cols:
            self.encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            )
            self.encoder.fit(df_tmp[self.cat_cols])

        if self.do_scaling and self.num_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(df_tmp[self.num_cols])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe using already-fitted encoder/scaler.
        Use this on TRAIN (after fit) and TEST.
        """
        df_out = self.add_engineered_features(df)

        if self.cat_cols:
            df_out = self._encode_categorical(df_out)

        if self.do_scaling and self.num_cols:
            df_out = self._scale_numeric(df_out)

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience: fit + transform in one step (like sklearn).
        """
        self.fit(df)
        return self.transform(df)
