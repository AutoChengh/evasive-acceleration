"""Native convex-cell stage of EA with compiled continuous-safety predicates.

No compilation is performed by a measured call. The library must be built in
advance with -fno-fast-math -ffp-contract=off. Every unresolved native result
continues through EA's certified angular-refinement stage inside the same call.
Internal reference/fallback labels describe these numerical stages, not a
different EA algorithm.
"""
import ctypes
import math
import os
import threading
from pathlib import Path
import numpy as np
from numba import cfunc,carray,types
from . import continuous as motion,priority_safety as safety
from . import penalty_safety as initial_safety,continuous_compact_kernel as proposal_safety
from .continuous_resilient_cells import CellBudget,planes
from .circle_proposal import propose,inside_polygon
from .normalized_refinement import refine_brent as refine_cut
from .._model.config import EAConfig
from .._model.four_modes import FOUR_PAIRS

_dp=types.CPointer(types.float64);_ip=types.CPointer(types.int64)
_cert_sig=types.intc(_dp,*([types.float64]*6),types.int64,_dp,_ip)


@cfunc(_cert_sig,cache=True)
def _initial(ptr,wa,wb,ax,ay,horizon,guard,limit,witness,nodes):
    status,t,n=initial_safety.certify(carray(ptr,(14,)),wa,wb,ax,ay,horizon,guard,limit)
    witness[0]=t;nodes[0]=n;return status


@cfunc(_cert_sig,cache=True)
def _proposal(ptr,wa,wb,ax,ay,horizon,guard,limit,witness,nodes):
    status,t,n=proposal_safety.certify(carray(ptr,(14,)),wa,wb,ax,ay,horizon,guard,limit)
    witness[0]=t;nodes[0]=n;return status


@cfunc(_cert_sig,cache=True)
def _main(ptr,wa,wb,ax,ay,horizon,guard,limit,witness,nodes):
    status,t,n=safety.certify(carray(ptr,(14,)),wa,wb,ax,ay,horizon,guard,limit)
    witness[0]=t;nodes[0]=n;return status


@cfunc(types.void(_dp,types.float64,types.float64,types.float64,types.float64,_dp),cache=True)
def _planes(ptr,wa,wb,t,guard,out):
    n,b=planes(carray(ptr,(14,)),wa,wb,t,guard)
    for j in range(8):out[2*j]=n[j,0];out[2*j+1]=n[j,1];out[16+j]=b[j]


@cfunc(types.float64(_dp,*([types.float64]*6)),cache=True)
def _refine(ptr,wa,wb,ax,ay,t,horizon):
    return refine_cut(carray(ptr,(14,)),wa,wb,ax,ay,t,horizon)


@cfunc(types.float64(_dp,*([types.float64]*5)),cache=True)
def _separation(ptr,wa,wb,t,ax,ay):
    return motion.separation(carray(ptr,(14,)),wa,wb,t,ax,ay)


@cfunc(types.intc(types.float64,_dp,_dp,_ip,types.int64,_dp),cache=True)
def _circle(radius,normals,offsets,counts,npolys,out):
    x,y,ok=propose(radius,carray(normals,(2048,8,2)),carray(offsets,(2048,8)),carray(counts,(2048,)),npolys)
    out[0]=x;out[1]=y;return int(ok)


@cfunc(types.intc(types.float64,_dp,types.int64,_dp),cache=True)
def _inside(radius,vertices,count,out):
    x,y,ok=inside_polygon(radius,carray(vertices,(count,2)),count)
    out[0]=x;out[1]=y;return int(ok)


_callbacks=(_initial,_proposal,_main,_planes,_refine,_separation,_circle,_inside)
_default_library=Path(__file__).resolve().parents[3]/'_native'/'libea_cells.so'
_library_path=Path(os.environ.get('EA_NATIVE_CELLS_LIBRARY',str(_default_library)))
if not _library_path.is_file():
    raise RuntimeError('Build native_cells.cpp in advance; expected library: '+str(_library_path))
_library=ctypes.CDLL(str(_library_path))
_library.ea_cells_native_abi.argtypes=[];_library.ea_cells_native_abi.restype=ctypes.c_int
if _library.ea_cells_native_abi()!=1:raise RuntimeError('Native EA ABI mismatch')
_native=_library.ea_cells_native_solve
_native.argtypes=[ctypes.POINTER(ctypes.c_double)]+[ctypes.c_double]*6+[ctypes.c_int]*4+[ctypes.c_void_p]*8+[ctypes.POINTER(ctypes.c_double)]
_native.restype=None
_library.ea_cells_native_seed_abi.argtypes=[];_library.ea_cells_native_seed_abi.restype=ctypes.c_int
if _library.ea_cells_native_seed_abi()!=1:raise RuntimeError('Native EA seed ABI mismatch')
_seed_create=_library.ea_cells_native_seed_create
_seed_create.argtypes=[ctypes.POINTER(ctypes.c_double)]+[ctypes.c_double]*4;_seed_create.restype=ctypes.c_void_p
_seed_destroy=_library.ea_cells_native_seed_destroy
_seed_destroy.argtypes=[ctypes.c_void_p];_seed_destroy.restype=None
_seeded=_library.ea_cells_native_solve_seeded
_seeded.argtypes=_native.argtypes[:-1]+[ctypes.c_void_p,_native.argtypes[-1]];_seeded.restype=None
_addresses=tuple(ctypes.c_void_p(f.address)for f in _callbacks)


class SeedContext:
    """Private within-call time/upper-proposal hints, never an EA answer cache."""
    def __init__(self,state_values,config=EAConfig(),budget=CellBudget()):
        s=np.ascontiguousarray(state_values,dtype=np.float64)
        if s.shape!=(14,)or not np.isfinite(s).all():raise ValueError('Fourteen finite state values required')
        self.pointer=_seed_create(s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),config.prediction_horizon_s,
            config.max_extra_acceleration_mps2,budget.acceleration_tolerance,budget.floating_guard_m)
        if not self.pointer:raise MemoryError('Cannot allocate native EA hint context')
        self.owner=threading.get_ident()
    def close(self):
        if self.pointer:
            if self.owner!=threading.get_ident():raise ValueError('EA hint contexts cannot be shared across threads')
            _seed_destroy(self.pointer);self.pointer=None
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def solve_pair(state_values,pair,config=EAConfig(),budget=CellBudget(),_seed=None):
    state=np.asarray(state_values,dtype=np.float64).copy()
    if state.shape!=(14,)or not np.isfinite(state).all():raise ValueError('Fourteen finite state values required')
    if any(state[i]<=0. for i in (4,5,11,12)):raise ValueError('Positive dimensions required')
    if len(pair)!=2 or any(int(v)not in (0,1)for v in pair):raise ValueError('CV/CTRV pairs required')
    if not(0.<config.prediction_horizon_s<math.inf and 0.<=config.max_extra_acceleration_mps2<math.inf):
        raise ValueError('Require positive finite horizon and nonnegative finite acceleration cap')
    if budget.max_vertices>128:raise ValueError('EA native cell capacity is 128 vertices')
    state[2],state[9]=max(0.,state[2]),max(0.,state[9])
    wa=state[6]if pair[0]else 0.;wb=state[13]if pair[1]else 0.
    wa=0. if abs(wa)<1e-6 else wa;wb=0. if abs(wb)<1e-6 else wb
    from . import continuous_parallel as preserved
    if wa==0. and wb==0.:return preserved.solve_pair(state,pair,config)
    out=np.empty(12,dtype=np.float64)
    arguments=(state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),wa,wb,
        config.prediction_horizon_s,config.max_extra_acceleration_mps2,budget.acceleration_tolerance,
        budget.floating_guard_m,budget.max_nodes,budget.max_vertices,budget.max_iterations,
        budget.max_certificate_nodes,*_addresses)
    if _seed is None:_native(*arguments,out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    else:
        if not _seed.pointer:raise ValueError('EA hint context is closed')
        if _seed.owner!=threading.get_ident():raise ValueError('EA hint contexts cannot be shared across threads')
        _seeded(*arguments,_seed.pointer,out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    result=safety.ContinuousResult(float(out[0]),float(out[1]),float(out[2]),float(out[3]),bool(out[4]),bool(out[5]),
        int(out[6]),int(out[7]),int(out[8]),int(out[9]),int(out[10]),
        ('certified-zero','certified-native-continuous','unused','current-contact-or-uncertainty',
         'unused','native-cell-unresolved','unused','certified-cap-infeasible')[int(out[11])])
    if not(result.continuous_verified and result.angular_converged)and result.status=='native-cell-unresolved':
        fallback=preserved.solve_pair(state,pair,config,preserved.RetainedBudget(
            acceleration_tolerance=budget.acceleration_tolerance,floating_guard_m=budget.floating_guard_m,
            clearance_m=max(1e-7,8.*budget.floating_guard_m)))
        if fallback.continuous_verified and fallback.angular_converged:
            return safety.ContinuousResult(fallback.value,fallback.direction_x,fallback.direction_y,
                fallback.lower_bound,True,True,result.iterations+fallback.iterations,
                result.time_constraints+fallback.time_constraints,result.direction_evaluations+fallback.direction_evaluations,
                result.sector_nodes+fallback.sector_nodes,result.certificate_nodes+fallback.certificate_nodes,
                'certified-native-reference-fallback')
        if fallback.status in ('current-contact-or-uncertainty','certified-cap-infeasible'):return fallback
    return result


def compute_selected_modes_with_diagnostics(state_values,config=EAConfig(),pairs=FOUR_PAIRS,budget=CellBudget()):
    results={};cache={}
    with SeedContext(state_values,config,budget)as seed:
        for pair in pairs:
            key=(float(state_values[6])if pair[0]else 0.,float(state_values[13])if pair[1]else 0.)
            key=tuple(0. if abs(w)<1e-6 else w for w in key)
            if key not in cache:cache[key]=solve_pair(state_values,pair,config,budget,_seed=seed)
            results[pair]=cache[key]
    return results


def compute_selected_modes(state_values,config=EAConfig(),pairs=FOUR_PAIRS,budget=CellBudget()):
    result=compute_selected_modes_with_diagnostics(state_values,config,pairs,budget)
    if any(not(r.continuous_verified and r.angular_converged)and r.status not in ('current-contact-or-uncertainty','certified-cap-infeasible')for r in result.values()):
        raise RuntimeError('Native result unresolved: no unsafe or inaccurate scalar is returned')
    return {pair:r.value for pair,r in result.items()}
