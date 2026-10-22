"""A cut heuristic must improve only proposals, never claim safety."""
import math
import numpy as np
from evasive_acceleration._experimental.normalized_refinement import refine,refine_brent
from evasive_acceleration._experimental.continuous import separation


def test_normalized_cut_does_not_weaken_a_real_witness():
    rng=np.random.default_rng(20261022)
    tested=0
    for _ in range(300):
        s=np.array([0.,0.,rng.uniform(0.,5.),rng.uniform(-3.,3.),4.5,1.8,rng.uniform(-32.,32.),
            rng.uniform(-4.,4.),rng.uniform(-4.,4.),rng.uniform(0.,5.),rng.uniform(-3.,3.),4.,1.7,rng.uniform(-32.,32.)])
        t=rng.uniform(.01,4.);ax,ay=rng.uniform(-4.,4.,2)
        gap=separation(s,s[6],s[13],t,ax,ay)
        if gap>=-1e-8:continue
        for oracle in (refine,refine_brent):
            before=s.copy();actual=oracle(s,s[6],s[13],ax,ay,t,10.)
            assert 0.<actual<=10.
            assert 2.*separation(s,s[6],s[13],actual,ax,ay)/(actual*actual)<=2.*gap/(t*t)+1e-10
            assert np.array_equal(s,before)
        tested+=1
    assert tested>30


def test_invalid_time_remains_a_non_certificate():
    s=np.array([0.,0.,1.,0.,4.,2.,0.,8.,0.,1.,0.,4.,2.,0.])
    for oracle in (refine,refine_brent):
        assert oracle(s,0.,0.,0.,0.,0.,10.)==0.
        assert oracle(s,0.,0.,0.,0.,11.,10.)==11.
        assert math.isnan(oracle(s,0.,0.,0.,0.,math.nan,10.))
