from wine_quality.features import FEATURES, TARGET
from wine_quality.prepare import load_raw, split


def test_load_raw_merges_both_wine_types():
    df = load_raw()
    assert list(df.columns) == FEATURES + [TARGET]
    assert set(df["is_red"].unique()) == {0, 1}
    assert len(df) == 1599 + 4898


def test_split_is_deterministic_and_stratified():
    df = load_raw()
    a_train, a_test = split(df, 0.2, seed=1)
    b_train, b_test = split(df, 0.2, seed=1)
    assert a_test.index.equals(b_test.index)
    assert len(a_train) + len(a_test) == len(df)
    # Every quality grade present in the data appears in the test split.
    assert set(a_test[TARGET]) == set(df[TARGET])
