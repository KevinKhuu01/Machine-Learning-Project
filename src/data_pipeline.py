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
    df["time_start"] = pd.to_datetime(df["time_start"], errors="raise")
    df["time_end"] = pd.to_datetime(df["time_end"], errors="raise")

    # Check for missing values in essential columns
    required_columns = [
        "count_id", "direction", "time_start", "time_end",
        "volume_15min", "longitude", "latitude"
    ]

    if df[required_columns].isna().any().any():
        raise ValueError("Missing values found in required columns.")

    # Check for duplicate observations
    duplicates = df.duplicated(
        subset=["count_id", "direction", "time_start"]
    )

    if duplicates.any():
        raise ValueError(
            f"Found {duplicates.sum()} duplicate observations."
        )

    # Verify that each observation represents 15 minutes
    durations = df["time_end"] - df["time_start"]

    if not (durations == pd.Timedelta(minutes=15)).all():
        raise ValueError("Found observations with invalid durations.")

    # Ensure traffic volumes are valid
    if (df["volume_15min"] < 0).any():
        raise ValueError("Found negative traffic volumes.")

    # Sort observations chronologically within each sequence
    df = df.sort_values(
        by=["count_id", "direction", "time_start"]
    ).reset_index(drop=True)

    # Check for gaps between consecutive observations
    time_differences = (
        df.groupby(["count_id", "direction"])["time_start"]
        .diff()
    )

    invalid_gaps = (
        time_differences.notna()
        & (time_differences != pd.Timedelta(minutes=15))
    )

    if invalid_gaps.any():
        raise ValueError(
            f"Found {invalid_gaps.sum()} gaps in traffic observations."
        )

    print("Preprocessing validation passed.")
    print(f"Total observations: {len(df)}")
    print(f"Count sessions: {df['count_id'].nunique()}")

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
# Creates day-ahead forecasting examples from consecutive calendar days.
#
# Each example uses 32 traffic counts from 9 AM to 5 PM on one day to
# predict the 32 corresponding counts on the following day. Only complete
# daily sequences from the same count session and direction are paired.
#
# Parameters:
#   df - Preprocessed traffic DataFrame.
#
# Returns:
#   DataFrame containing previous-day features and next-day targets.
# ---------------------------------------------------------------------------
def create_day_ahead_dataset(df):
    df = df.copy()

    # Restrict observations to the forecasting window
    df = df[
        (df["time_start"].dt.hour >= 9)
        & (df["time_start"].dt.hour < 17)
    ].copy()

    df["date"] = df["time_start"].dt.normalize()

    # Organize complete daily traffic sequences
    daily_sequences = {}

    for (count_id, direction, date), group in df.groupby(
        ["count_id", "direction", "date"]
    ):
        group = group.sort_values("time_start")

        # A complete daytime window contains 32 intervals
        expected_times = pd.date_range(
            start=date + pd.Timedelta(hours=9),
            periods=32,
            freq="15min"
        )

        if (
            len(group) != 32
            or not group["time_start"].reset_index(drop=True).equals(
                pd.Series(expected_times)
            )
        ):
            continue

        daily_sequences[(count_id, direction, date)] = group

    # Create pairs of consecutive days
    examples = []

    for (count_id, direction, date), current_day in daily_sequences.items():
        next_date = date + pd.Timedelta(days=1)
        next_day = daily_sequences.get(
            (count_id, direction, next_date)
        )

        if next_day is None:
            continue

        example = {
            "count_id": count_id,
            "direction": direction,
            "input_date": date,
            "target_date": next_date,
            "latitude": current_day["latitude"].iloc[0],
            "longitude": current_day["longitude"].iloc[0],
            "target_day_of_week": next_date.dayofweek,
        }

        # Previous day's traffic volumes become features
        for i, volume in enumerate(current_day["volume_15min"]):
            example[f"previous_volume_{i}"] = volume

        # Following day's traffic volumes become targets
        for i, volume in enumerate(next_day["volume_15min"]):
            example[f"target_volume_{i}"] = volume

        examples.append(example)

    return pd.DataFrame(examples)



# ---------------------------------------------------------------------------
# Splits day-ahead examples by geographic location.
#
# Unique locations are randomly ordered using a fixed seed and allocated
# approximately 70% to training, 15% to validation, and 15% to testing.
# All examples at the same location remain in the same split.
#
# Parameters:
#   df           - Day-ahead forecasting DataFrame.
#   random_state - Seed used for reproducible location splitting.
#
# Returns:
#   train_df, valid_df, test_df
# ---------------------------------------------------------------------------
def split_day_ahead_data(df, random_state=42):

    # Identify unique locations
    locations = (
        df[["latitude", "longitude"]]
        .drop_duplicates()
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    # Allocate approximately 70/15/15 of locations
    train_end = int(len(locations) * 0.70)
    valid_end = int(len(locations) * 0.85)

    train_locations = locations.iloc[:train_end]
    valid_locations = locations.iloc[train_end:valid_end]
    test_locations = locations.iloc[valid_end:]

    # Assign each example to its location's split
    location_columns = ["latitude", "longitude"]

    train_df = df.merge(train_locations, on=location_columns)
    valid_df = df.merge(valid_locations, on=location_columns)
    test_df = df.merge(test_locations, on=location_columns)

    return train_df, valid_df, test_df



# ---------------------------------------------------------------------------
# Selects features and targets for day-ahead forecasting.
#
# Each input contains 32 previous-day volumes, the target day of week,
# coordinates, and encoded direction. Each target contains 32 traffic
# volumes for the following day.
#
# Parameters:
#   train_df - Training examples.
#   valid_df - Validation examples.
#   test_df  - Test examples.
#
# Returns:
#   Feature matrices (X) and 32-column target matrices (y) for each split.
# ---------------------------------------------------------------------------
def prepare_day_ahead_model_data(train_df, valid_df, test_df):

    # Previous-day traffic volumes
    volume_features = [
        f"previous_volume_{i}" for i in range(32)
    ]

    # Additional information available before forecasting
    other_features = [
        "target_day_of_week",
        "latitude",
        "longitude"
    ]

    # Encode traffic direction consistently across all splits
    direction_categories = ["EB", "WB", "NB", "SB"]

    def prepare_features(df):
        X = df[volume_features + other_features].copy()

        for direction in direction_categories:
            X[f"direction_{direction}"] = (
                df["direction"] == direction
            ).astype(int)

        return X

    # Create feature matrices
    X_train = prepare_features(train_df)
    X_valid = prepare_features(valid_df)
    X_test = prepare_features(test_df)

    # Next-day traffic volumes
    target_columns = [
        f"target_volume_{i}" for i in range(32)
    ]

    y_train = train_df[target_columns].copy()
    y_valid = valid_df[target_columns].copy()
    y_test = test_df[target_columns].copy()

    return X_train, X_valid, X_test, y_train, y_valid, y_test

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


# ---------------------------------------------------------------------------
# Runs the complete day-ahead forecasting data pipeline.
#
# Loads and validates the raw observations, pairs consecutive daytime
# sequences, splits examples by location, and prepares model-ready data.
#
# Returns:
#   X_train, X_valid, X_test, y_train, y_valid, y_test
# ---------------------------------------------------------------------------
def get_day_ahead_model_data():

    # Load and validate the raw observations
    df = load_data()
    df = preprocess_data(df)

    # Create consecutive-day forecasting examples
    day_ahead_df = create_day_ahead_dataset(df)

    # Split examples by geographic location
    train_df, valid_df, test_df = split_day_ahead_data(
        day_ahead_df
    )

    # Prepare model features and targets
    return prepare_day_ahead_model_data(
        train_df,
        valid_df,
        test_df
    )


# ---------------------------------------------------------------------------
# Checks both forecasting pipelines when the file is run directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nOriginal 15-minute forecasting pipeline:")
    original_data = get_model_data()

    print("\nDay-ahead forecasting pipeline:")
    day_ahead_data = get_day_ahead_model_data()

    for label, datasets in [
        ("Original", original_data),
        ("Day-ahead", day_ahead_data),
    ]:
        X_train, X_valid, X_test, y_train, y_valid, y_test = datasets
        print(f"\n{label} model-ready datasets:")
        for name, X, y in [
            ("Training", X_train, y_train),
            ("Validation", X_valid, y_valid),
            ("Test", X_test, y_test),
        ]:
            print(f"{name}: X={X.shape}, y={y.shape}")
            assert not X.isna().any().any(), f"Missing {name} features"
            assert not y.isna().any().any(), f"Missing {name} targets"
        assert X_train.columns.equals(X_valid.columns)
        assert X_train.columns.equals(X_test.columns)

    # Validate the day-ahead example construction and location separation.
    day_ahead_df = create_day_ahead_dataset(preprocess_data(load_data()))
    assert (
        day_ahead_df["target_date"] - day_ahead_df["input_date"]
        == pd.Timedelta(days=1)
    ).all(), "Day-ahead targets must be one calendar day later"
    assert set(day_ahead_df["direction"]).issubset(
        {"EB", "WB", "NB", "SB"}
    ), "Unexpected traffic direction"

    train_df, valid_df, test_df = split_day_ahead_data(day_ahead_df)
    location_sets = [
        set(zip(part["latitude"], part["longitude"]))
        for part in (train_df, valid_df, test_df)
    ]
    assert not (location_sets[0] & location_sets[1])
    assert not (location_sets[0] & location_sets[2])
    assert not (location_sets[1] & location_sets[2])

    print("\nAll pipeline checks passed.")
