from __future__ import annotations

import pandas as pd
import pytest

from gatecheck import data

from .conftest import make_players


def test_read_raw_normalises_string_booleans(tmp_path):
    path = tmp_path / "players.csv"
    path.write_text(
        "userid,version,sum_gamerounds,retention_1,retention_7\n"
        "1,gate_30,3,False,False\n"
        "2,gate_40,38,TRUE,false\n",
        encoding="utf-8",
    )
    frame = data.read_raw(path)
    assert frame["retention_1"].dtype == bool
    assert frame["retention_1"].tolist() == [False, True]
    assert frame["sum_gamerounds"].tolist() == [3, 38]


def test_read_raw_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("userid,version\n1,gate_30\n", encoding="utf-8")
    with pytest.raises(data.SchemaError, match="columns missing"):
        data.read_raw(path)


def test_read_raw_rejects_unknown_boolean_token(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "userid,version,sum_gamerounds,retention_1,retention_7\n1,gate_30,3,maybe,False\n",
        encoding="utf-8",
    )
    with pytest.raises(data.SchemaError, match="unexpected boolean"):
        data.read_raw(path)


def test_read_raw_rejects_negative_rounds(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "userid,version,sum_gamerounds,retention_1,retention_7\n1,gate_30,-3,True,False\n",
        encoding="utf-8",
    )
    with pytest.raises(data.SchemaError, match="negative"):
        data.read_raw(path)


def test_missing_file_points_at_the_provenance_note(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        data.read_raw(tmp_path / "nope.csv")


def test_parquet_cache_round_trip(tmp_path, players):
    csv_path = tmp_path / "players.csv"
    players.to_csv(csv_path, index=False)
    cache = tmp_path / "players.parquet"
    first = data.load_players(csv_path, cache)
    assert cache.exists()
    second = data.load_players(csv_path, cache)
    pd.testing.assert_frame_equal(first, second)


def test_split_arms_separates_cleanly(players):
    split = data.split_arms(players, "gate_30", "gate_40")
    assert split.n_control == 1_200
    assert split.n_treatment == 1_200
    assert set(split.control["version"].unique()) == {"gate_30"}


def test_split_arms_rejects_a_third_label():
    frame = make_players(n_control=10, n_treatment=10)
    frame.loc[0, "version"] = "gate_50"
    with pytest.raises(data.SchemaError, match="unexpected arm labels"):
        data.split_arms(frame, "gate_30", "gate_40")


def test_split_arms_rejects_a_missing_arm(players):
    only_control = players.loc[players["version"] == "gate_30"]
    with pytest.raises(data.SchemaError, match="expected both"):
        data.split_arms(only_control, "gate_30", "gate_40")


def test_winsorise_clips_at_the_reported_cut(players):
    clipped, cut = data.winsorise(players["sum_gamerounds"], 0.90)
    assert clipped.max() <= cut
    assert cut <= players["sum_gamerounds"].max()


def test_winsorise_rejects_a_low_quantile(players):
    with pytest.raises(ValueError, match="0.5, 1.0"):
        data.winsorise(players["sum_gamerounds"], 0.25)
