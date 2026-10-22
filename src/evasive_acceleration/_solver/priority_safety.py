"""EA best-bound-first rectangle certificate with exact contact rejections.

Fixed-axis Taylor lower bounds cover the whole horizon. Traversal
prioritizes the interval with the lowest bound, so a grazing interval cannot
starve a deeper collision elsewhere. Exhaustion is unresolved, never safe.
"""
import math
import numpy as np
from .continuous import njit,_kinematics,_axis_gap,_linearized_gap_bound,separation
from .continuous_compact_kernel import (
    ContinuousResult,ContinuousBudget,solve_pair,_store_time,_ray,
    sector_lower_bound,_refine_witness_time,_full_certify,
)
from .continuous_tail_kernel import acceleration_tail_start


@njit(cache=True,fastmath=False)
def _interval(state,wa,wb,ax,ay,lo,hi,trans_second,second,radius,guard):
    mid,half=.5*(lo+hi),.5*(hi-lo)
    dx,dy,vx,vy,ca,sa,cb,sb=_kinematics(state,wa,wb,mid,ax,ay)
    circle=math.hypot(dx,dy)-math.hypot(vx,vy)*half-.5*trans_second*half*half-radius
    if circle>guard:return circle,circle
    # Fixed-axis bounds certify separation, not the numerical value of the
    # rotating SAT-axis score when disjoint. For penetrating boxes their
    # negative parts are valid lower bounds, sufficient for rejection priority
    # and pruning below a known NEGATIVE witness.
    lower=circle;gap=-np.inf
    for j in range(4):
        nx=ca if j==0 else -sa if j==1 else cb if j==2 else -sb
        ny=sa if j==0 else ca if j==1 else sb if j==2 else cb
        gap=max(gap,_axis_gap(state,dx,dy,ca,sa,cb,sb,nx,ny))
        lower=max(lower,_linearized_gap_bound(state,dx,dy,vx,vy,ca,sa,cb,sb,wa,wb,nx,ny,half,second))
    return lower,gap


@njit(cache=True,fastmath=False)
def _push(bounds,left,right,size,bound,lo,hi):
    at=size;size+=1
    while at:
        parent=(at-1)//2
        if bounds[parent]<=bound:break
        bounds[at],left[at],right[at]=bounds[parent],left[parent],right[parent];at=parent
    bounds[at],left[at],right[at]=bound,lo,hi
    return size


@njit(cache=True,fastmath=False)
def _pop(bounds,left,right,size):
    bound,lo,hi=bounds[0],left[0],right[0];size-=1
    if size:
        b,l,r=bounds[size],left[size],right[size];at=0
        while 2*at+1<size:
            child=2*at+1
            if child+1<size and bounds[child+1]<bounds[child]:child+=1
            if bounds[child]>=b:break
            bounds[at],left[at],right[at]=bounds[child],left[child],right[child];at=child
        bounds[at],left[at],right[at]=b,l,r
    return bound,lo,hi,size


@njit(cache=True,fastmath=False)
def certify(state,wa,wb,ax,ay,horizon,guard=1e-10,max_nodes=30000):
    horizon=min(horizon,acceleration_tail_start(state,ax,ay,guard))
    ra,rb=.5*math.hypot(state[4],state[5]),.5*math.hypot(state[11],state[12])
    trans_second=state[2]*abs(wa)+state[9]*abs(wb)+math.hypot(ax,ay)
    second=trans_second+ra*wa*wa+rb*wb*wb
    best_gap=0.;best_time=np.nan;unresolved=np.nan;negative_started=0
    for t in (0.,horizon):
        gap=separation(state,wa,wb,t,ax,ay)
        if gap< -guard:
            best_gap,best_time=gap,t
            if gap< -1e-3 or t==0.:return 0,t,1
    capacity=min(8192,max(8,max_nodes+2))
    bounds,left,right=np.empty(capacity),np.empty(capacity),np.empty(capacity)
    lower,gap=_interval(state,wa,wb,ax,ay,0.,horizon,trans_second,second,ra+rb,guard)
    visited=1
    if gap< -guard and gap<best_gap:best_gap,best_time=gap,.5*horizon
    if gap< -1e-3:return 0,.5*horizon,visited
    size=_push(bounds,left,right,0,lower,0.,horizon)
    while size:
        bound,lo,hi,size=_pop(bounds,left,right,size)
        if bound>guard:continue
        if math.isfinite(best_time):
            if bound>=best_gap:continue
            if not negative_started:negative_started=visited
            if visited-negative_started>=512:return 0,best_time,visited
        mid=.5*(lo+hi)
        if .5*(hi-lo)<1e-11:
            if not math.isfinite(unresolved):unresolved=mid
            continue
        if visited+2>max_nodes or size+2>=capacity:
            return (0,best_time,visited)if math.isfinite(best_time)else(-1,mid,visited)
        for a,b in ((lo,mid),(mid,hi)):
            lower,gap=_interval(state,wa,wb,ax,ay,a,b,trans_second,second,ra+rb,guard);visited+=1
            if gap< -guard and gap<best_gap:
                best_gap,best_time=gap,.5*(a+b)
                if gap< -1e-3:return 0,best_time,visited
            if lower<=guard:size=_push(bounds,left,right,size,lower,a,b)
    if math.isfinite(best_time):return 0,best_time,visited
    if math.isfinite(unresolved):return -1,unresolved,visited
    return 1,np.nan,visited
