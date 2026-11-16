"""Stage 1: merge red and white datasets and create a stratified train/test split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from wine_quality.config import PROCESSED_DIR, RAW_DIR, load_params
from wine_quality.features import FEATURES, TARGET


def load_raw() -> pd.DataFrame:
    red = pd.read_csv(RAW_DIR / "winequality_red.csv").assign(is_red=1)
    white = pd.read_csv(RAW_DIR / "winequality_white.csv").assign(is_red=0)
    df = pd.concat([red, white], ignore_index=True)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Raw data contains missing values")
    return df[FEATURES + [TARGET]]


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Stratify on quality so rare grades (3 and 9) appear in both splits.
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])


def main() -> None:
    params = load_params()
    df = load_raw()
    train, test = split(df, params["prepare"]["test_size"], params["seed"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"prepare: {len(df)} rows -> train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
