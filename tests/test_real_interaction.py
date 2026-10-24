"""Real-data showcases reuse EA unchanged and protect inputs and old runs."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ea_real_interaction_example',ROOT/'examples/real_interaction.py')
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)


@pytest.fixture(scope='module')
def recorded():
    return pd.read_csv(ROOT/'demo_data'/example.CASES['crossing']['input'])


def test_real_case_recomputes_the_bundled_states(recorded):
    result=example.evaluate_frames(recorded)
    assert len(result)==195 and np.isfinite(result.EA_m_s2).all()
    assert result.EA_m_s2.max()==pytest.approx(1.19148879674147,abs=.001)
    assert result['Elapsed time (s)'].iloc[-1]==pytest.approx(7.76)
    assert result.EA_m_s2.iloc[0]==0 and result.EA_m_s2.iloc[-1]==0
    pd.testing.assert_frame_equal(result[recorded.columns],recorded)


def test_display_window_does_not_trim_exported_observations(recorded):
    result=example.evaluate_frames(recorded)
    assert example.figure_window(result)==pytest.approx((0.,3.))
    assert len(result)==len(recorded)==195


def test_future_changes_do_not_change_earlier_ea(recorded):
    first=example.evaluate_frames(recorded.iloc[:20])
    changed=recorded.copy()
    changed.loc[changed.index[20:],'2_Position X (m)']+=100
    second=example.evaluate_frames(changed)
    np.testing.assert_array_equal(first.EA_m_s2,second.EA_m_s2.iloc[:20])


@pytest.mark.parametrize('mutation',[
    lambda d:d.drop(columns=['Yawrate']),lambda d:d.iloc[::-1],lambda d:d.iloc[:0],
    lambda d:d.assign(**{'Time (s)':0}),lambda d:d.assign(Heading=np.nan),
    lambda d:d.assign(**{'Vehicle Number':np.arange(len(d))}),
])
def test_invalid_pair_or_state_order_rejected(recorded,mutation):
    with pytest.raises(ValueError):example.evaluate_frames(mutation(recorded.iloc[:4].copy()))


def test_exports_full_real_results_with_unchanged_settings(tmp_path):
    output=tmp_path/'fresh'
    result=example.run_real_cases(output,case='all',make_plot=False)
    metadata=json.loads((output/'run.json').read_text())
    assert {key:len(value) for key,value in result.items()}=={'crossing':195,'low-speed':208}
    for case,value in metadata['cases'].items():
        assert value['synthetic'] is False and value['future_observations_used_for_ea'] is False
        assert value['yaw_rate_source']=='provided'
        assert value['solver_settings']=={'a_max':100.0,'tol':.001,'T_total':10.0,'size_scale':1.0}
        assert len(value['input_sha256'])==64
        assert len(pd.read_csv(output/f'{case}_results.csv'))==value['observations']
    assert metadata['cases']['low-speed']['peak_ea_m_s2']==pytest.approx(.20233395111987185,abs=.001)


def test_input_and_prior_output_cannot_be_overwritten(tmp_path):
    output=tmp_path/'existing';output.mkdir();marker=output/'original.csv';marker.write_bytes(b'preserved')
    with pytest.raises(FileExistsError):example.run_real_cases(output,make_plot=False)
    assert marker.read_bytes()==b'preserved'
    source=ROOT/'demo_data'/example.CASES['crossing']['input'];before=source.read_bytes()
    with pytest.raises(FileExistsError):example.run_real_cases(source,make_plot=False)
    assert source.read_bytes()==before


def test_plot_failure_creates_no_output_directory(tmp_path,monkeypatch):
    def broken(*args,**kwargs):raise RuntimeError('plot unavailable')
    monkeypatch.setattr(example,'plot_case',broken)
    from evasive_acceleration import plotting
    monkeypatch.setattr(plotting,'resolve_figure_font',lambda **kwargs:{'actual_family':'Arial'})
    output=tmp_path/'new'
    with pytest.raises(RuntimeError,match='plot unavailable'):example.run_real_cases(output)
    assert not output.exists()


def test_shared_figure_style_is_regular_and_no_control_arrow_claim(recorded):
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    figure=example.plot_case(example.evaluate_frames(recorded),allow_font_fallback=True)
    try:
        figure.canvas.draw()
        for text in figure.findobj(Text):
            assert text.get_fontweight() in ('normal',400)
        assert 'EA (m/s²)' in [text.get_text() for text in figure.findobj(Text)]
        assert 'Arrows indicate heading' in (ROOT/'docs/real_interactions.md').read_text(encoding='utf-8')
    finally:plt.close(figure)


def test_homepage_leads_with_real_data_not_feature_inventory():
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    assert 'Start with the question you want to answer' not in readme
    assert 'Equal-TTC comparison' not in readme and 'two manual controls' not in readme
    assert 'Both cases have TTC = 2.00' not in readme
    assert '## Interactive notebook' not in readme
    assert 'python examples/real_interaction.py' in readme
    assert readme.index('## Visual examples')<readme.index('## Run a real interaction')<readme.index('## Install and run')
    assert (ROOT/'assets/figures/ea_real_interaction.png').is_file()
    assert (ROOT/'assets/figures/ea_real_low_speed.png').is_file()
