import math
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_compute import process_one_csv
from single_frame import RoadUserState, compute_single_frame_ea


def test_single_frame_demo_value_is_finite():
    agent_a = RoadUserState(0.0, 0.0, 10.0, 0.0, 4.5, 1.8, 0.0)
    agent_b = RoadUserState(20.0, 0.0, 8.0, math.pi, 4.7, 1.9, 0.0)

    value = compute_single_frame_ea(agent_a, agent_b)

    assert math.isfinite(value)
    assert value > 0.0


def test_batch_demo_writes_expected_columns(tmp_path):
    import pandas as pd

    src_csv = PROJECT_ROOT / "demo_data" / "InD_18_tracks_417_424.csv"
    input_csv = tmp_path / src_csv.name
    output_dir = tmp_path / "batch_results"
    pd.read_csv(src_csv).head(3).to_csv(input_csv, index=False)

    stat = process_one_csv(str(input_csv), str(output_dir))
    out_csv = Path(stat["output_file"])

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    for col in ["EA_CVCV", "EA_CVCT", "EA_CTCV", "EA_CTCT", "EA"]:
        assert col in df.columns
    assert df["EA"].notna().any()
