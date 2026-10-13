from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "svc_raw_data_volume_2020_2024.csv"


# ---------------------------------------------------------------------------
# Loads the raw 15-minute traffic volume dataset from the project's data
# directory.
#
# Returns:
#   DataFrame containing the raw traffic observations.
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

# ---------------------------------------------------------------------------
# Prepares the raw traffic data for target and feature creation.
#
# Converts the timestamp columns to datetime objects and sorts observations
# chronologically within each count session and direction. Keeping the
# observations in chronological order is necessary when creating the
# next-interval prediction target.
#
# Parameters:
#   df - Raw traffic DataFrame.
#
# Returns:
#   Preprocessed and chronologically sorted DataFrame.
# ---------------------------------------------------------------------------
def preprocess_data(df):
    df = df.copy()

    # Convert timestamps to datetime objects
    df["time_start"] = pd.to_datetime(df["time_start"])
    df["time_end"] = pd.to_datetime(df["time_end"])

    # Sort observations chronologically within each traffic count
    df = df.sort_values(
        by=["count_id", "direction", "time_start"]
    ).reset_index(drop=True)

    return df

# ---------------------------------------------------------------------------
# Creates the target variable for the traffic forecasting problem.
#
# The target, next_volume, represents the traffic volume observed during the
# next 15-minute interval for the same count session and direction. The last
# observation in each sequence is removed because there is no following
# interval available to use as its target.
#
# Parameters:
#   df - Preprocessed traffic DataFrame.
#
# Returns:
#   DataFrame containing the next_volume target.
# ---------------------------------------------------------------------------
def create_target(df):
    df = df.copy()

    # Traffic volume in the next 15-minute interval
    df["next_volume"] = (
        df.groupby(["count_id", "direction"])["volume_15min"]
        .shift(-1)
    )

    # Remove the final observation from each sequence because
    # there is no following 15-minute interval to predict
    df = df.dropna(subset=["next_volume"]).copy()

    df["next_volume"] = df["next_volume"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Creates time-based features that can be used by the baseline model.
#
# Hour, minute, and day of week are extracted from the starting timestamp
# to provide the model with information about when each traffic observation
# occurred.
#
# Parameters:
#   df - Traffic DataFrame containing datetime information.
#
# Returns:
#   DataFrame containing the additional time-based features.
# ---------------------------------------------------------------------------
def create_features(df):
    df = df.copy()

    # Time-based features
    df["hour"] = df["time_start"].dt.hour
    df["minute"] = df["time_start"].dt.minute
    df["day_of_week"] = df["time_start"].dt.dayofweek

    return df

# ---------------------------------------------------------------------------
# Splits the dataset into training, validation, and test sets.
#
# Count sessions are first ordered according to their starting time. Complete
# sessions are then assigned approximately 70% to training, 15% to validation,
# and 15% to testing. Keeping an entire count_id in one split prevents the
# same traffic-count session from appearing in multiple datasets.
#
# Parameters:
#   df - Processed traffic DataFrame.
#
# Returns:
#   train_df, valid_df, test_df
# ---------------------------------------------------------------------------
def split_data(df):
    # Find the start time of each count session
    sessions = (
        df.groupby("count_id")["time_start"]
        .min()
        .sort_values()
    )

    session_ids = sessions.index.tolist()

    train_end = int(len(session_ids) * 0.70)
    valid_end = int(len(session_ids) * 0.85)

    train_ids = session_ids[:train_end]
    valid_ids = session_ids[train_end:valid_end]
    test_ids = session_ids[valid_end:]

    train_df = df[df["count_id"].isin(train_ids)].copy()
    valid_df = df[df["count_id"].isin(valid_ids)].copy()
    test_df = df[df["count_id"].isin(test_ids)].copy()

    return train_df, valid_df, test_df

# ---------------------------------------------------------------------------
# Selects the features and target that will be passed to the baseline model.
#
# The initial feature set contains the current traffic volume, time-based
# features, and geographic coordinates. Identifier columns are excluded.
# The prediction target is next_volume.
#
# Parameters:
#   train_df - Training observations.
#   valid_df - Validation observations.
#   test_df  - Test observations.
#
# Returns:
#   Feature matrices (X) and target vectors (y) for all three datasets.
# ---------------------------------------------------------------------------
def prepare_model_data(train_df, valid_df, test_df):
    features = [
        "volume_15min",
        "hour",
        "minute",
        "day_of_week",
        "longitude",
        "latitude",
    ]

    target = "next_volume"

    X_train = train_df[features].copy()
    y_train = train_df[target].copy()

    X_valid = valid_df[features].copy()
    y_valid = valid_df[target].copy()

    X_test = test_df[features].copy()
    y_test = test_df[target].copy()

    return X_train, X_valid, X_test, y_train, y_valid, y_test

# ---------------------------------------------------------------------------
# Runs the complete data pipeline.
#
# Loads the raw dataset, preprocesses it, creates the prediction target and
# features, performs the train/validation/test split, and returns the
# model-ready datasets. Other project files can call this function without
# needing to repeat the individual pipeline steps.
#
# Returns:
#   X_train, X_valid, X_test, y_train, y_valid, y_test
# ---------------------------------------------------------------------------
def get_model_data():
    df = load_data()
    df = preprocess_data(df)
    df = create_target(df)
    df = create_features(df)

    train_df, valid_df, test_df = split_data(df)

    return prepare_model_data(train_df, valid_df, test_df)

if __name__ == "__main__":
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_model_data()

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    print("\nMissing values:")
    print("Train:", X_train.isna().sum().sum(), y_train.isna().sum())
    print("Validation:", X_valid.isna().sum().sum(), y_valid.isna().sum())
    print("Test:", X_test.isna().sum().sum(), y_test.isna().sum())