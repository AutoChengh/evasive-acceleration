"""EA preserves the public API, infinite contact states and causal input path."""

import hashlib
import json
import math
from pathlib import Path
import runpy

import pandas as pd
import pytest

from batch_compute import process_one_csv
from core_ea import CORE_ID
from evasive_acceleration import (
    RoadUserKinematicState, RoadUserState, build_compute_kwargs,
    build_state_with_estimated_yaw_rate, compute_ea,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("state_class, values", [
    (RoadUserState, (0, 0, -1, 0, 4.5, 1.8, 0)),
    (RoadUserKinematicState, (0, 0, -1, 0, 4.5, 1.8)),
])
def test_public_state_rejects_negative_speed_like_batch(state_class, values):
    with pytest.raises(ValueError, match="non-negative"):
        state_class(*values).validate()


@pytest.mark.parametrize("option", [
    "T_total", "dt_coarse", "dt_fine", "a_max", "tol", "size_scale",
    "local_fine_half_window_deg", "fine_refine_ratio",
])
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_nonfinite_options_fail_before_entering_the_solver(option, value):
    state = RoadUserState(0, 0, 5, 0, 4.5, 1.8, 0)
    with pytest.raises(ValueError, match="finite"):
        build_compute_kwargs(state, state, **{option: value})


def test_causal_yaw_and_provided_yaw_use_the_same_ea_solver():
    current = RoadUserKinematicState(0, 0, 10, 0.02, 4.5, 1.8)
    estimated, diagnostic = build_state_with_estimated_yaw_rate(
        current, [0, 0.1, 0.2], [0, 0.01, 0.02],
    )
    provided = RoadUserState(0, 0, 10, 0.02, 4.5, 1.8, 0.1)
    actor = RoadUserState(20, 0, 8, math.pi, 4.7, 1.9, 0)
    assert diagnostic.initialized and estimated.yaw_rate == 0.1
    assert compute_ea(estimated, actor) == compute_ea(provided, actor)


def test_batch_preserves_infinite_contact_and_records_core_and_input(tmp_path):
    source = tmp_path / "contact.csv"
    frame = pd.DataFrame({
        "Position X (m)": [0.0], "Position Y (m)": [0.0],
        "Velocity (m/s)": [5.0], "Heading": [0.0],
        "Length (m)": [4.5], "Width (m)": [1.8], "Yawrate": [0.0],
        "2_Position X (m)": [0.0], "2_Position Y (m)": [0.0],
        "2_Velocity (m/s)": [2.0], "2_Heading": [0.0],
        "2_Length (m)": [4.5], "2_Width (m)": [1.8], "2_Yawrate": [0.0],
    })
    frame.to_csv(source, index=False)
    before = source.read_bytes()
    stat = process_one_csv(
        str(source), str(tmp_path / "out"),
        yaw_rate_source="provided", skip_baselines=True,
    )
    result = pd.read_csv(stat["output_file"])
    metadata = json.loads(Path(stat["metadata_file"]).read_text(encoding="utf-8"))
    for column in ("EA", "EA_CVCV", "EA_CVCT", "EA_CTCV", "EA_CTCT"):
        assert result.loc[0, column] == math.inf
    assert result.loc[0, "EA_status"] == "ok"
    assert stat["computed_success_frames"] == 1
    assert stat["computation_failed_frames"] == 0
    assert metadata["core_id"] == CORE_ID
    assert metadata["input_sha256"] == hashlib.sha256(before).hexdigest()
    assert metadata["dependency_versions"]["numpy"]
    assert source.read_bytes() == before


def test_gif_colors_distinguish_unavailable_zero_and_capped_infinite_ea():
    pytest.importorskip("matplotlib")
    renderer = runpy.run_path(str(ROOT / "visualization/visualize_tracks_to_gif.py"))
    mapper, _ = renderer["build_ea_color_mapper"](1.5)
    assert mapper(math.nan) != mapper(0.0)
    assert mapper(None) == mapper(math.nan)
    assert mapper(math.inf) == mapper(1.5)
    assert mapper(math.inf) != mapper(0.0)
