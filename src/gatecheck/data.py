"""Loading, schema checks and the parquet cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ("userid", "version", "sum_gamerounds", "retention_1", "retention_7")


class SchemaError(ValueError):
    """Raised when the raw file is not the file we think it is."""


@dataclass(frozen=True)
class ArmSplit:
    """The two arms of the experiment, already separated."""

    control_name: str
    treatment_name: str
    control: pd.DataFrame
    treatment: pd.DataFrame

    @property
    def n_control(self) -> int:
        return len(self.control)

    @property
    def n_treatment(self) -> int:
        return len(self.treatment)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = set(lowered.unique()) - set(mapping)
    if unknown:
        raise SchemaError(f"unexpected boolean values: {sorted(unknown)}")
    return lowered.map(mapping).astype(bool)


def read_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV and normalise dtypes.

    The file ships booleans as the strings ``True``/``False``, which pandas will
    read as ``object`` on some versions and ``bool`` on others, so the cast is
    explicit rather than left to chance.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. See data/README.md for where the file comes from."
        )
    frame = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"columns missing from {path.name}: {missing}")
    frame = frame.loc[:, list(EXPECTED_COLUMNS)].copy()
    frame["userid"] = frame["userid"].astype("int64")
    frame["version"] = frame["version"].astype("string")
    frame["sum_gamerounds"] = frame["sum_gamerounds"].astype("int64")
    for column in ("retention_1", "retention_7"):
        frame[column] = _coerce_bool(frame[column])
    if (frame["sum_gamerounds"] < 0).any():
        raise SchemaError("sum_gamerounds contains negative values")
    return frame


def cache_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Write the normalised frame to parquet so repeat runs skip the CSV parse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
    return path


def load_players(raw_csv: Path, parquet_cache: Path | None = None) -> pd.DataFrame:
    """Load the player table, using the parquet cache when it is fresh."""
    cache_is_fresh = (
        parquet_cache is not None
        and parquet_cache.exists()
        and parquet_cache.stat().st_mtime >= raw_csv.stat().st_mtime
    )
    if cache_is_fresh:
        return pd.read_parquet(parquet_cache, engine="pyarrow")
    frame = read_raw(raw_csv)
    if parquet_cache is not None:
        cache_parquet(frame, parquet_cache)
    return frame


def split_arms(frame: pd.DataFrame, control: str, treatment: str) -> ArmSplit:
    """Separate the two arms, refusing anything with a third label in it."""
    labels = set(frame["version"].dropna().unique())
    unexpected = labels - {control, treatment}
    if unexpected:
        raise SchemaError(f"unexpected arm labels: {sorted(unexpected)}")
    if not {control, treatment} <= labels:
        raise SchemaError(f"expected both {control} and {treatment}, found {sorted(labels)}")
    return ArmSplit(
        control_name=control,
        treatment_name=treatment,
        control=frame.loc[frame["version"] == control].reset_index(drop=True),
        treatment=frame.loc[frame["version"] == treatment].reset_index(drop=True),
    )


def winsorise(values: pd.Series, upper_quantile: float) -> tuple[pd.Series, float]:
    """Clip the top tail and hand back the cut point that was used."""
    if not 0.5 < upper_quantile <= 1.0:
        raise ValueError("upper_quantile must sit in (0.5, 1.0]")
    cut = float(values.quantile(upper_quantile))
    return values.clip(upper=cut), cut
