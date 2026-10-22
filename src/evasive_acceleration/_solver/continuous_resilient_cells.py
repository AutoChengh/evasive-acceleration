"""EA convex-cell refinement with certified angular completion.

At a real collision time every feasible acceleration must escape one of the
eight SAT halfplanes. Partition the current convex cell into those escape
pieces and keep their exact min-norm bounds. The frontier carries these bounds
through successive contact constraints. No unresolved scalar is accepted;
upper proposals still need full continuous certification. Internal
reference/fallback labels denote EA angular refinement, not another algorithm.
"""
from dataclasses import dataclass
import math
import numpy as np
from . import continuous as motion,priority_safety as safety
from . import penalty_safety as initial_safety,continuous_compact_kernel as proposal_safety
from .circle_proposal import inside_polygon,propose as circle_proposal
from .continuous_compact_kernel import _refine_witness_time
from .._model.config import EAConfig
from .._model.four_modes import FOUR_PAIRS
njit=motion.njit


@dataclass(frozen=True)
class CellBudget:
    acceleration_tolerance: float=.001
    floating_guard_m: float=1e-9
    max_nodes: int=4096
    max_vertices: int=96
    max_iterations: int=8192
    max_certificate_nodes: int=30000

    def __post_init__(self):
        if not(math.isfinite(self.acceleration_tolerance)and self.acceleration_tolerance>0.):
            raise ValueError('Require positive finite acceleration tolerance')
        if not(math.isfinite(self.floating_guard_m)and self.floating_guard_m>0.):
            raise ValueError('Require positive finite floating guard')
        for value in (self.max_nodes,self.max_vertices,self.max_iterations,self.max_certificate_nodes):
            if not isinstance(value,int)or isinstance(value,bool)or value<1:
                raise ValueError('Search budgets must be positive integers')
        if self.max_vertices<12:raise ValueError('Require at least twelve cell vertices')


@njit(cache=True,fastmath=False)
def clip(vertices,count,nx,ny,bound,out):
    size=0
    for j in range(count):
        i=count-1 if j==0 else j-1
        px,py=vertices[i,0],vertices[i,1];qx,qy=vertices[j,0],vertices[j,1]
        fp,fq=nx*px+ny*py-bound,nx*qx+ny*qy-bound
        pin,qin=fp<=0.,fq<=0.
        if pin!=qin:
            alpha=fp/(fp-fq);out[size,0],out[size,1]=px+alpha*(qx-px),py+alpha*(qy-py);size+=1
        if qin:out[size,0],out[size,1]=qx,qy;size+=1
    return size


@njit(cache=True,fastmath=False)
def nearest(vertices,count):
    inside=True;best,bx,by=np.inf,np.nan,np.nan
    for j in range(count):
        k=(j+1)%count;px,py=vertices[j,0],vertices[j,1];qx,qy=vertices[k,0],vertices[k,1]
        dx,dy=qx-px,qy-py
        if dx*(-py)-dy*(-px)<0.:inside=False
        squared=dx*dx+dy*dy
        alpha=max(0.,min(1.,-(px*dx+py*dy)/squared))if squared else 0.
        x,y=px+alpha*dx,py+alpha*dy;r=math.hypot(x,y)
        if r<best:best,bx,by=r,x,y
    return (0.,0.,0.)if inside else(best,bx,by)


@njit(cache=True,fastmath=False)
def planes(state,wa,wb,t,guard):
    dx,dy,vx,vy,ca,sa,cb,sb=motion._kinematics(state,wa,wb,t,0.,0.)
    n=np.empty((8,2));b=np.empty(8)
    for j in range(4):
        nx=ca if j==0 else -sa if j==1 else cb if j==2 else -sb
        ny=sa if j==0 else ca if j==1 else sb if j==2 else cb
        radius=.5*(state[4]*abs(nx*ca+ny*sa)+state[5]*abs(-nx*sa+ny*ca)
            +state[11]*abs(nx*cb+ny*sb)+state[12]*abs(-nx*sb+ny*cb))-guard
        for k in range(2):
            sign=1. if k==0 else -1.;i=2*j+k
            n[i,0],n[i,1]=sign*nx,sign*ny
            b[i]=2.*(sign*(nx*dx+ny*dy)+radius)/(t*t)
    return n,b


@njit(cache=True,fastmath=False)
def _store(state,wa,wb,t,guard,times,normals,offsets,count):
    if t<=0.:return count
    for k in range(count):
        if abs(times[k]-t)<1e-12:return count
    if count>=len(times):return count
    n,b=planes(state,wa,wb,t,guard);times[count]=t
    for j in range(8):
        normals[count,j,0],normals[count,j,1]=n[j,0],n[j,1];offsets[count,j]=b[j]
    return count+1


@njit(cache=True,fastmath=False)
def _cached(x,y,normals,offsets,count,err):
    best=-err;at=-1
    for k in range(count):
        gap=-np.inf
        for j in range(8):
            gap=max(gap,normals[k,j,0]*x+normals[k,j,1]*y-offsets[k,j])
            if gap>=best:break
        if gap<best:best,at=gap,k
    return at


@njit(cache=True,fastmath=False)
def _solve(state,wa,wb,horizon,cap,tolerance,guard,max_nodes,max_vertices,max_iterations,cert_limit):
    status,witness,cn=initial_safety.certify(state,wa,wb,0.,0.,horizon,guard,cert_limit)
    if status==1:return 0.,np.nan,np.nan,0.,True,True,0,0,0,0,cn,0
    if motion.separation(state,wa,wb,0.,0.,0.)<=guard:
        return np.inf,np.nan,np.nan,cap,False,False,0,0,0,0,cn,3
    vertices=np.empty((max_nodes,max_vertices,2));counts=np.empty(max_nodes,np.int64);bounds=np.empty(max_nodes)
    vertices[0,0,0],vertices[0,0,1]=-cap,-cap
    vertices[0,1,0],vertices[0,1,1]=cap,-cap
    vertices[0,2,0],vertices[0,2,1]=cap,cap
    vertices[0,3,0],vertices[0,3,1]=-cap,cap
    counts[0]=4;bounds[0]=0.;pending=1;best,bx,by=np.inf,np.nan,np.nan
    cell=np.empty((max_vertices+2,2));work=np.empty((max_vertices+2,2));piece=np.empty((max_vertices+2,2))
    lower=0.;nodes=1;cuts=0
    err=2048.*np.finfo(np.float64).eps*(cap+1.)
    times=np.empty(2048);pool_n=np.empty((2048,8,2));pool_b=np.empty((2048,8));pool_counts=np.full(2048,8,np.int64)
    pool=0
    for fraction in (.25,.5,.75,1.,1.25,1.5,2.):
        pool=_store(state,wa,wb,min(horizon,witness*fraction),guard,times,pool_n,pool_b,pool)
    for fraction in (.25,.5,.75,1.):pool=_store(state,wa,wb,horizon*fraction,guard,times,pool_n,pool_b,pool)
    fast_rotation=max(abs(wa),abs(wb))>5.
    initial=256 if fast_rotation else 24 if max(abs(wa),abs(wb))>.1 else 8
    for j in range(initial+1):
        t=min(horizon,witness*((.75+.5*j/initial)if fast_rotation else(.25+2.*j/initial)))
        pool=_store(state,wa,wb,t,guard,times,pool_n,pool_b,pool)
    circle_period=1 if .1<max(abs(wa),abs(wb))<=5. else 64
    for iteration in range(max_iterations):
        if pending==0:
            if math.isfinite(best):return best,bx,by,max(0.,best-tolerance),True,True,iteration,cuts,0,nodes,cn,1
            return np.inf,np.nan,np.nan,cap,False,True,iteration,cuts,0,nodes,cn,7
        at=np.argmin(bounds[:pending]);lower=bounds[at]
        if math.isfinite(best)and best-lower<=tolerance:
            return best,bx,by,max(0.,best-tolerance),True,True,iteration,cuts,0,nodes,cn,1
        count=counts[at]
        for j in range(count):cell[j,0],cell[j,1]=vertices[at,j,0],vertices[at,j,1]
        pending-=1
        if at<pending:
            bounds[at],counts[at]=bounds[pending],counts[pending]
            for j in range(counts[at]):vertices[at,j,0],vertices[at,j,1]=vertices[pending,j,0],vertices[pending,j,1]
        radius,x,y=nearest(cell,count)
        if radius-err>cap:continue
        known=_cached(x,y,pool_n,pool_b,pool,err)
        if known<0:
            for fraction in (.9,.25):
                qx,qy,ok=0.,0.,False
                if iteration%circle_period==0:qx,qy,ok=circle_proposal(radius+fraction*tolerance,pool_n,pool_b,pool_counts,pool)
                if not ok:qx,qy,ok=inside_polygon(radius+fraction*tolerance,cell,count)
                if not ok:continue
                value=math.hypot(qx,qy)
                if value>cap or value>=best or _cached(qx,qy,pool_n,pool_b,pool,err)>=0:continue
                check,t,visited=proposal_safety.certify(state,wa,wb,qx,qy,horizon,guard,cert_limit);cn+=visited
                if check==1:best,bx,by=value,qx/value if value else np.nan,qy/value if value else np.nan
                elif check==0 and t>0.:
                    refined=_refine_witness_time(state,wa,wb,qx,qy,t,horizon)
                    if refined>0. and motion.separation(state,wa,wb,refined,qx,qy)<motion.separation(state,wa,wb,t,qx,qy):t=refined
                    pool=_store(state,wa,wb,t,guard,times,pool_n,pool_b,pool)
                if math.isfinite(best)and best-max(0.,radius-err)<=tolerance:break
            if math.isfinite(best)and best-max(0.,radius-err)<=tolerance:continue
            known=_cached(x,y,pool_n,pool_b,pool,err)
        if known<0:
            check,t,visited=safety.certify(state,wa,wb,x,y,horizon,guard,cert_limit);cn+=visited
            if check==1:
                if radius<best:best,bx,by=radius,x/radius if radius else np.nan,y/radius if radius else np.nan
                continue
            if check!=0 or t<=0.:
                return best,bx,by,lower,math.isfinite(best),False,iteration+1,cuts,0,nodes,cn,5
            refined=_refine_witness_time(state,wa,wb,x,y,t,horizon)
            if refined>0. and motion.separation(state,wa,wb,refined,x,y)<motion.separation(state,wa,wb,t,x,y):t=refined
            normal,offset=planes(state,wa,wb,t,guard)
            pool=_store(state,wa,wb,t,guard,times,pool_n,pool_b,pool)
        else:normal,offset=pool_n[known],pool_b[known]
        strict=True
        for j in range(8):
            if normal[j,0]*x+normal[j,1]*y>=offset[j]-err:strict=False;break
        if not strict:
            return best,bx,by,lower,math.isfinite(best),False,iteration+1,cuts,0,nodes,cn,5
        cuts+=1
        # Disjoint escape partition: outside face j AND inside all earlier
        # faces. Their union covers every point outside this inner collider.
        remaining=count
        for j in range(8):
            if remaining<3:break
            nx,ny,b=normal[j,0],normal[j,1],offset[j]
            size=clip(cell,remaining,-nx,-ny,-b,piece)
            if size>=3:
                r,qx,qy=nearest(piece,size);bound=max(lower,max(0.,r-err))
                if bound<=cap and(not math.isfinite(best)or best-bound>tolerance):
                    if pending>=max_nodes or size>max_vertices:
                        return best,bx,by,lower,math.isfinite(best),False,iteration+1,cuts,0,nodes,cn,5
                    counts[pending],bounds[pending]=size,bound
                    for k in range(size):vertices[pending,k,0],vertices[pending,k,1]=piece[k,0],piece[k,1]
                    pending+=1;nodes+=1
            size=clip(cell,remaining,nx,ny,b,work)
            if size>max_vertices:
                return best,bx,by,lower,math.isfinite(best),False,iteration+1,cuts,0,nodes,cn,5
            remaining=size
            for k in range(size):cell[k,0],cell[k,1]=work[k,0],work[k,1]
    return best,bx,by,lower,math.isfinite(best),False,max_iterations,cuts,0,nodes,cn,5


def solve_pair(state_values,pair,config=EAConfig(),budget=CellBudget()):
    state=np.asarray(state_values,dtype=np.float64).copy()
    if state.shape!=(14,)or not np.isfinite(state).all():raise ValueError('Fourteen finite state values required')
    if any(state[i]<=0. for i in (4,5,11,12)):raise ValueError('Positive dimensions required')
    if len(pair)!=2 or any(int(v)not in (0,1)for v in pair):raise ValueError('CV/CTRV pairs required')
    if not(0.<config.prediction_horizon_s<math.inf and 0.<=config.max_extra_acceleration_mps2<math.inf):
        raise ValueError('Require positive finite horizon and nonnegative finite acceleration cap')
    state[2],state[9]=max(0.,state[2]),max(0.,state[9])
    wa=state[6]if pair[0]else 0.;wb=state[13]if pair[1]else 0.
    wa=0. if abs(wa)<1e-6 else wa;wb=0. if abs(wb)<1e-6 else wb
    if wa==0. and wb==0.:
        from .continuous_parallel import solve_pair as preserved
        return preserved(state,pair,config)
    r=_solve(state,wa,wb,config.prediction_horizon_s,config.max_extra_acceleration_mps2,
        budget.acceleration_tolerance,budget.floating_guard_m,budget.max_nodes,budget.max_vertices,
        budget.max_iterations,budget.max_certificate_nodes)
    result=safety.ContinuousResult(*r[:11],('certified-zero','certified-continuous','unused',
        'current-contact-or-uncertainty','unused','cell-unresolved','unused','certified-cap-infeasible')[r[11]])
    if not(result.continuous_verified and result.angular_converged)and result.status=='cell-unresolved':
        from . import continuous_parallel as preserved
        fallback=preserved.solve_pair(state,pair,config,preserved.RetainedBudget(
            acceleration_tolerance=budget.acceleration_tolerance,
            floating_guard_m=budget.floating_guard_m,
            clearance_m=max(1e-7,8.*budget.floating_guard_m)))
        if fallback.continuous_verified and fallback.angular_converged:
            return safety.ContinuousResult(fallback.value,fallback.direction_x,fallback.direction_y,
                fallback.lower_bound,True,True,result.iterations+fallback.iterations,
                result.time_constraints+fallback.time_constraints,result.direction_evaluations+fallback.direction_evaluations,
                result.sector_nodes+fallback.sector_nodes,result.certificate_nodes+fallback.certificate_nodes,
                'certified-cell-reference-fallback')
        if fallback.status in ('current-contact-or-uncertainty','certified-cap-infeasible'):return fallback
    return result


def compute_selected_modes_with_diagnostics(state_values,config=EAConfig(),pairs=FOUR_PAIRS,budget=CellBudget()):
    results={};cache={}
    for pair in pairs:
        key=(float(state_values[6])if pair[0]else 0.,float(state_values[13])if pair[1]else 0.)
        key=tuple(0. if abs(w)<1e-6 else w for w in key)
        if key not in cache:cache[key]=solve_pair(state_values,pair,config,budget)
        results[pair]=cache[key]
    return results


def compute_selected_modes(state_values,config=EAConfig(),pairs=FOUR_PAIRS,budget=CellBudget()):
    result=compute_selected_modes_with_diagnostics(state_values,config,pairs,budget)
    if any(not(r.continuous_verified and r.angular_converged)and r.status not in ('current-contact-or-uncertainty','certified-cap-infeasible')for r in result.values()):
        raise RuntimeError('Cell result unresolved: no unsafe or inaccurate scalar is returned')
    return {pair:r.value for pair,r in result.items()}
