import math
import numpy as np
from evasive_acceleration._experimental import circle_proposal as p


def _outside(x,y,normals,offsets,counts):
    return all(max(normals[k,j,0]*x+normals[k,j,1]*y-offsets[k,j]
        for j in range(counts[k]))>=-1e-12 for k in range(len(counts)))


def test_circle_gap_finds_an_off_grid_narrow_interval():
    angle=.123456;width=1e-5;c,s=math.cos(angle),math.sin(angle)
    normals=np.array([[[c,s],[-c,-s],[-s,c],[s,-c]]])
    offsets=np.array([[math.cos(width),2.,2.,2.]])
    x,y,ok=p.propose(1.,normals,offsets,np.array([4]),1)
    assert ok and abs(math.hypot(x,y)-1.)<1e-12
    assert c*x+s*y>offsets[0,0]


def test_fully_covered_circle_has_no_candidate():
    normals=np.array([[[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]])
    offsets=np.array([[2.,2.,2.,2.]])
    x,y,ok=p.propose(1.,normals,offsets,np.array([4]),1)
    assert not ok


def test_polygon_arc_union_matches_independent_point_predicate():
    rng=np.random.default_rng(20261022)
    theta=np.linspace(0.,2.*math.pi,1001)
    for _ in range(40):
        normals=np.empty((8,4,2));offsets=np.empty((8,4));counts=np.full(8,4)
        for k in range(8):
            a=rng.uniform(-math.pi,math.pi);c,s=math.cos(a),math.sin(a)
            normals[k]=[[c,s],[-c,-s],[-s,c],[s,-c]]
            center=rng.uniform(-2.,2.,2);r=rng.uniform(.1,1.2,2)
            offsets[k]=[np.dot(normals[k,j],center)+r[j//2]for j in range(4)]
        x,y,ok=p.propose(1.,normals,offsets,counts,8)
        if ok:assert _outside(x,y,normals,offsets,counts)
        else:assert not any(_outside(math.cos(a),math.sin(a),normals,offsets,counts)for a in theta)
