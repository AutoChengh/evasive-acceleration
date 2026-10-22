import math
import numpy as np
from evasive_acceleration._solver import priority_safety as p
from evasive_acceleration._solver import continuous as reference


def test_heap_preserves_all_intervals_in_lower_bound_order():
    rng=np.random.default_rng(44);values=rng.normal(size=100)
    bounds,left,right=np.empty(102),np.empty(102),np.empty(102);size=0
    for i,v in enumerate(values):size=p._push(bounds,left,right,size,v,float(i),float(i+1))
    actual=[];indices=[]
    while size:
        b,l,r,size=p._pop(bounds,left,right,size);actual.append(b);indices.append(int(l))
        assert r==l+1.
    np.testing.assert_array_equal(actual,np.sort(values))
    assert sorted(indices)==list(range(100))


def test_priority_interval_bound_is_conservative_against_independent_times():
    rng=np.random.default_rng(73)
    for _ in range(35):
        s=np.array([0.,0.,rng.uniform(0.,8.),rng.uniform(-math.pi,math.pi),4.4,1.9,rng.uniform(-3.,3.),
            rng.uniform(-15.,15.),rng.uniform(-15.,15.),rng.uniform(0.,8.),rng.uniform(-math.pi,math.pi),4.,1.8,rng.uniform(-3.,3.)])
        ax,ay=rng.uniform(-5.,5.,2);lo=rng.uniform(0.,9.);hi=lo+rng.uniform(.001,1.)
        ra=.5*math.hypot(s[4],s[5]);rb=.5*math.hypot(s[11],s[12])
        trans=s[2]*abs(s[6])+s[9]*abs(s[13])+math.hypot(ax,ay)
        second=trans+ra*s[6]**2+rb*s[13]**2
        bound,gap=p._interval(s,s[6],s[13],ax,ay,lo,hi,trans,second,ra+rb,1e-9)
        true=min(reference.separation(s,s[6],s[13],t,ax,ay)for t in np.linspace(lo,hi,201))
        # A fixed separating axis need not be a box's CURRENT SAT axis when
        # the boxes are disjoint. Its positive gap can exceed that particular
        # SAT score; their negative parts, and the collision sign, are the
        # comparable certificate quantities.
        assert min(bound,0.)<=min(true,0.)+1e-10
        if bound>1e-9:assert true>0.


def test_priority_certificate_keeps_between_sample_collisions():
    state=np.array([0.,0.,0.,0.,.2,.2,0.,.6,0.,20.,math.pi,.2,.2,0.])
    status,t,n=p.certify(state,0.,0.,.0001,0.,1.,1e-9,30000)
    assert status==0 and reference.separation(state,0.,0.,t,.0001,0.)< -1e-9


def test_priority_safe_has_full_reference_certificate():
    state=np.array([0.,0.,1.,0.,4.,2.,.01,0.,30.,2.,0.,4.,2.,.02])
    status,t,n=p.certify(state,state[6],state[13],.1,0.,10.,1e-9,30000)
    assert status==1
    assert reference.certify(state,state[6],state[13],.1,0.,10.,1e-9,30000)[0]==1


def test_priority_budget_exhaustion_is_not_a_safe_result():
    state=np.array([0.,0.,1.,0.,4.,2.,.01,4.01,0.,1.,0.,4.,2.,.02])
    status,t,n=p.certify(state,state[6],state[13],.1,0.,10.,1e-9,1)
    assert status!=1
