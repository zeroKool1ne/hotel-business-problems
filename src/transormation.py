import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class HotelTransformer:
    """
    Transformation pipeline for hotel booking data.

    This class applies a series of preprocessing steps, including feature
    engineering, one-hot encoding of categorical variables, and optional
    scaling of numerical columns. The class follows an sklearn-like API
    with fit, transform, and fit_transform methods.

    Args:
        cat_cols (list[str], optional): List of categorical column names to encode.
        num_cols (list[str], optional): List of numerical column names to scale.
        do_scaling (bool, optional): Whether to apply StandardScaler to numeric
            columns. Defaults to True.
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
        Create additional engineered features for the dataset.

        This function adds simple, meaningful features that combine existing
        columns to enhance predictive modeling. Currently, the following
        engineered features are created:
            - "total_stay": Sum of weekday and weekend stays.
            - "total_guests": Total number of guests (adults + children + babies).

        Args:
            df (pandas.DataFrame): Input DataFrame containing the original
                hotel booking data.

        Returns:
            pandas.DataFrame: A new DataFrame containing all original columns
                plus the newly engineered feature columns.

        Raises:
            KeyError: If any of the required columns for feature creation
                are missing from the input DataFrame.
        """
       
        df["total_stay"] = (
            df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        )
        df["total_guests"] = (
            df["adults"] + df["children"].fillna(0) + df["babies"]
        )
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns using a fitted encoder.

        This function transforms the categorical variables into one-hot encoded
        vectors. The encoder must be fitted beforehand using the fit() or
        fit_transform() method.

        Args:
            df (pandas.DataFrame): The DataFrame containing categorical columns
                to be encoded.

        Returns:
            pandas.DataFrame: A DataFrame where categorical columns have been
                replaced by their corresponding one-hot encoded columns.

        Raises:
            RuntimeError: If the encoder has not been fitted before calling
                this method.
        """
        # If no categorical columns are defined, return the DataFrame unchanged.
        if not self.cat_cols:
            return df

        if self.encoder is None:
            raise RuntimeError("Encoder is not fitted. Call `fit` or `fit_transform` first.")
        # Transform the categorical columns into one-hot encoded arrays.
        encoded = self.encoder.transform(df[self.cat_cols])
        
        # Convert the encoded numpy array into a DataFrame with proper column names
        # and aligned index so it can be concatenated back into the original df.
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(self.cat_cols),
            index=df.index,
        )
        # Drop original categorical columns from the DataFrame.
        df = df.drop(columns=self.cat_cols)

        # Concatenate the encoded columns to the remaining DataFrame.
        df = pd.concat([df, encoded_df], axis=1)

        return df

    def _scale_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical columns using a fitted StandardScaler.

        This function applies z-score scaling (mean = 0, std = 1) to the
        numerical features. Scaling is optional and controlled by the
        do_scaling flag. The scaler must be fitted beforehand.

        Args:
            df (pandas.DataFrame): The DataFrame containing numerical
                columns to scale.

        Returns:
            pandas.DataFrame: A DataFrame where specified numerical columns
                have been standardized using the fitted scaler.

        Raises:
            RuntimeError: If scaling is enabled but the scaler has not
                been fitted yet.
        """

        # If scaling is disabled or there are no numerical columns defined,
        # return the DataFrame unchanged.
        if not self.do_scaling or not self.num_cols:
            return df

        # Ensure that the scaler has been fitted before applying transformations.
        # If not, the user likely called transform() before fit() or fit_transform().
        if self.scaler is None:
            raise RuntimeError("Scaler is not fitted. Call `fit` or `fit_transform`   first.")
        # Apply the fitted StandardScaler to the numerical columns.
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        
        return df

    # ---------- public API (sklearn-like) ----------

    def fit(self, df: pd.DataFrame):
        """
        Fit the encoder and scaler on the provided training DataFrame.

        This method prepares the transformation pipeline by fitting the
        OneHotEncoder on categorical columns and the StandardScaler on
        numerical columns. It should only be called on the training data.

        Args:
            df (pandas.DataFrame): Input training DataFrame used to learn
                categorical values and scaling statistics.

        Returns:
            HotelTransformer: The fitted transformer instance.
        """
        
        # First, apply feature engineering so that new columns are included
        # when fitting both the encoder and the scaler.
        df_tmp = self.add_engineered_features(df)

        
        # Fit OneHotEncoder on categorical columns (if any are provided).
        # This learns all unique categories and prepares the encoder for
        # transforming both train and test data consistently.
        if self.cat_cols:
            self.encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            )
            self.encoder.fit(df_tmp[self.cat_cols])
            
        # Fit StandardScaler on numerical columns (if scaling is enabled).
        # This learns the mean and standard deviation of each numeric feature
        # so future data can be standardized in the same way.
        if self.do_scaling and self.num_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(df_tmp[self.num_cols])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all fitted transformations to the input DataFrame.

        This method applies feature engineering, one-hot encoding
        (using the already-fitted encoder), and optional numeric scaling
        (using the already-fitted scaler). It can be used on both
        training and test sets after fitting.

        Args:
            df (pandas.DataFrame): The input DataFrame to transform.

        Returns:
            pandas.DataFrame: The fully transformed DataFrame containing
                engineered features, encoded categorical columns, and
                optionally scaled numeric columns.

        Raises:
            RuntimeError: If transform is called before fit has been run.
        """

        # Apply feature engineering first so that the new columns are included
        # in all subsequent transformation steps.
        df_out = self.add_engineered_features(df)

        # If categorical columns are defined, apply one-hot encoding using
        # the already-fitted encoder. This replaces all categorical fields
        # with their corresponding encoded vectors.
        if self.cat_cols:
            df_out = self._encode_categorical(df_out)

        # If numeric scaling is enabled and numerical columns exist,
        # standardize them using the fitted StandardScaler. This ensures
        # consistent scaling between train and test data.
        if self.do_scaling and self.num_cols:
            df_out = self._scale_numeric(df_out)

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the transformer and immediately apply the transformation.

        This is a convenience method that combines fit() and transform()
        into a single call, similar to sklearn's fit_transform() pattern.

        Args:
            df (pandas.DataFrame): The input DataFrame used to both fit
                the transformer and apply transformations.

        Returns:
            pandas.DataFrame: The transformed DataFrame with all preprocessing
                steps applied.
        """
        self.fit(df)
        return self.transform(df)
