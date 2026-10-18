"""The optional figures must not silently substitute publication typography."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from evasive_acceleration.plotting import (
    FIG1_COLORS, figure_style, plot_risk_history, resolve_figure_font,
)


def test_fig1_palette_and_regular_font_context():
    previous = list(matplotlib.rcParams["font.family"])
    with figure_style(allow_font_fallback=True) as font:
        assert matplotlib.rcParams["font.family"] == [font["actual_family"]]
        assert matplotlib.rcParams["font.weight"] == "normal"
        assert matplotlib.rcParams["axes.titleweight"] == "normal"
        assert FIG1_COLORS["green"] == "#81B199"
        assert FIG1_COLORS["gold"] == "#F1CB8D"
    assert matplotlib.rcParams["font.family"] == previous


def test_missing_arial_is_error_unless_substitution_explicit(monkeypatch):
    from matplotlib import font_manager

    monkeypatch.delenv("EA_ARIAL_FONT_PATH", raising=False)
    original = font_manager.findfont

    def without_arial(properties, *args, **kwargs):
        if "Arial" in properties.get_family():
            raise ValueError("Arial absent in test")
        return original(properties, *args, **kwargs)

    monkeypatch.setattr(font_manager, "findfont", without_arial)
    with pytest.raises(RuntimeError, match="Arial is required"):
        resolve_figure_font(allow_font_fallback=False)
    with pytest.warns(UserWarning, match="not Arial"):
        font = resolve_figure_font(allow_font_fallback=True)
    assert font["fallback_used"] is True


def test_custom_font_rejects_bold_arial(monkeypatch, tmp_path):
    from matplotlib import font_manager
    from types import SimpleNamespace

    path = tmp_path / "arial-bold.ttf"
    path.touch()
    monkeypatch.setenv("EA_ARIAL_FONT_PATH", str(path))
    monkeypatch.setattr(font_manager.fontManager, "addfont", lambda path: None)
    monkeypatch.setattr(font_manager.FontProperties, "get_name", lambda self: "Arial")
    monkeypatch.setattr(font_manager, "get_font", lambda path: SimpleNamespace(style_flags=2))
    with pytest.raises(ValueError, match="not bold or italic"):
        resolve_figure_font()


def test_history_uses_distinct_axes_and_does_not_mutate_data(tmp_path, monkeypatch):
    monkeypatch.setenv("EA_ALLOW_FONT_FALLBACK", "1")
    source = pd.DataFrame({"Time (s)": [5, 5.1, 5.2], "EA": [0.1, 0.3, 0.2],
                           "TTC": [2.0, float("inf"), 1.8]})
    original = source.copy(deep=True)
    output = tmp_path / "history.png"
    figure = plot_risk_history(source, output_path=output)
    try:
        assert len(figure.axes) == 2
        assert figure.axes[0].get_ylabel() == "EA (m/s\N{SUPERSCRIPT TWO})"
        assert figure.axes[1].get_ylabel() == "TTC (s)"
        assert output.is_file()
        for text in figure.findobj(matplotlib.text.Text):
            assert text.get_fontweight() in ("normal", 400)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for text in [figure.axes[0].title, figure.axes[1].xaxis.label,
                     *figure.texts]:
            extent = text.get_window_extent(renderer)
            assert extent.y0 >= 0
            assert extent.y1 <= figure.bbox.y1
        pd.testing.assert_frame_equal(source, original)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, float("nan")]])
def test_history_rejects_invalid_time_order(times):
    frame = pd.DataFrame({"Time (s)": times, "EA": [0, 0], "TTC": [1, 1]})
    with pytest.raises(ValueError, match="timestamps"):
        plot_risk_history(frame)
