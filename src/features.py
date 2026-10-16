"""Turn raw sensor readings into features a model can learn from."""


def add_rolling_features(df, sensors, window=5):
    """For each sensor, add the rolling mean and rolling std over the
    last `window` cycles of the SAME engine. This smooths out noise and
    captures trends, which single readings cannot show."""
    df = df.sort_values(["unit", "cycle"]).copy()
    grouped = df.groupby("unit")
    for s in sensors:
        df[f"{s}_mean{window}"] = grouped[s].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{s}_std{window}"] = grouped[s].transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
    return df


def get_feature_columns(df):
    """Every column except the engine ID and the target is a feature."""
    return [c for c in df.columns if c not in ("unit", "RUL")]