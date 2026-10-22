import math
import numpy as np
import pytest
from evasive_acceleration._experimental import continuous_resilient_cells as c
from evasive_acceleration._experimental import continuous_parallel as reference


def _state():return np.array([0.,0.,2.,0.,4.,2.,.1,8.,0.,1.,math.pi,4.,2.,.2])


@pytest.mark.parametrize('kwargs',[
    {'acceleration_tolerance':float('nan')},{'acceleration_tolerance':0.},
    {'floating_guard_m':float('inf')},{'max_nodes':0},{'max_vertices':4},
    {'max_iterations':1.5},{'max_certificate_nodes':True},
])
def test_invalid_resilient_budget_is_rejected(kwargs):
    with pytest.raises(ValueError):c.CellBudget(**kwargs)


def test_unresolved_primary_uses_only_a_certified_reference_result(monkeypatch):
    raw=(float('inf'),float('nan'),float('nan'),0.,False,False,1,1,0,1,1,5)
    monkeypatch.setattr(c,'_solve',lambda *args:raw)
    expected=c.safety.ContinuousResult(2.,1.,0.,1.999,True,True,7,4,8,9,10,'certified-continuous')
    monkeypatch.setattr(reference,'solve_pair',lambda *args:expected)
    result=c.solve_pair(_state(),(1,1))
    assert result.value==2. and result.continuous_verified and result.angular_converged
    assert result.status=='certified-cell-reference-fallback'and result.iterations==8


def test_two_unresolved_solvers_still_reject_scalar_output(monkeypatch):
    raw=(float('inf'),float('nan'),float('nan'),0.,False,False,1,1,0,1,1,5)
    monkeypatch.setattr(c,'_solve',lambda *args:raw)
    fallback=c.safety.ContinuousResult(float('inf'),float('nan'),float('nan'),0.,False,False,1,1,0,1,1,'unresolved')
    monkeypatch.setattr(reference,'solve_pair',lambda *args:fallback)
    with pytest.raises(RuntimeError,match='unresolved'):c.compute_selected_modes(_state(),pairs=((1,1),))


def test_real_reference_fallback_remains_within_same_precision():
    state=_state();result=c.solve_pair(state,(1,1),budget=c.CellBudget(max_iterations=1))
    expected=reference.solve_pair(state,(1,1))
    assert result.continuous_verified and result.angular_converged
    assert abs(result.value-expected.value)<=.001
    assert result.value-result.lower_bound<=.001+1e-12
