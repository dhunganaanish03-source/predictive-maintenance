"""Load the NASA C-MAPSS files and create Remaining Useful Life (RUL) labels."""
import pandas as pd

# The files have no header row, so we name the 26 columns ourselves.
COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125  # engines barely degrade early on, so we cap RUL at 125 cycles


def load_raw(path):
    """Read one space-separated C-MAPSS file into a DataFrame."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def add_train_rul(df):
    """In training data every engine runs until it fails,
    so RUL = (last cycle of that engine) - (current cycle)."""
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (last_cycle - df["cycle"]).clip(upper=RUL_CAP)
    return df


def add_test_rul(test_df, rul_path):
    """Test engines stop before failure. The RUL file gives the true RUL
    at each engine's LAST recorded cycle; we work backwards from it."""
    test_df = test_df.copy()
    true_rul = pd.read_csv(rul_path, header=None, names=["final_rul"])
    true_rul["unit"] = range(1, len(true_rul) + 1)
    test_df = test_df.merge(true_rul, on="unit")
    last_cycle = test_df.groupby("unit")["cycle"].transform("max")
    test_df["RUL"] = (test_df["final_rul"] + last_cycle - test_df["cycle"]).clip(upper=RUL_CAP)
    return test_df.drop(columns="final_rul")


def find_useless_columns(train_df, threshold=1e-4):
    """Sensors that never change carry no information about wear."""
    candidates = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    return [c for c in candidates if train_df[c].std() < threshold]