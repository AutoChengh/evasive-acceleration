"""EA exact-time cut proposals in acceleration units, never a safety test.

The SAT gap at t becomes an acceleration-space halfplane distance when
multiplied by 2/t**2. Locally improving that violation can produce stronger
necessary cuts than minimizing the unnormalized distance in metres. No
unimodality is assumed, and every upper bound still needs the full certifier.
"""
import math
from .continuous import njit,separation


@njit(cache=True,fastmath=False)
def refine(state,wa,wb,ax,ay,witness,horizon):
    if not(0.<witness<=horizon):return witness
    width=min(.05,horizon/200.)
    lo,hi=max(1e-12,witness-width),min(horizon,witness+width)
    best_time=witness
    best=2.*separation(state,wa,wb,witness,ax,ay)/(witness*witness)
    ratio=(math.sqrt(5.)-1.)/2.
    x=hi-ratio*(hi-lo);y=lo+ratio*(hi-lo)
    fx=2.*separation(state,wa,wb,x,ax,ay)/(x*x)
    fy=2.*separation(state,wa,wb,y,ax,ay)/(y*y)
    for _ in range(16):
        if fx<best:best,best_time=fx,x
        if fy<best:best,best_time=fy,y
        if fx<fy:
            hi,y,fy=y,x,fx;x=hi-ratio*(hi-lo)
            fx=2.*separation(state,wa,wb,x,ax,ay)/(x*x)
        else:
            lo,x,fx=x,y,fy;y=lo+ratio*(hi-lo)
            fy=2.*separation(state,wa,wb,y,ax,ay)/(y*y)
    if fx<best:best,best_time=fx,x
    if fy<best:best_time=y
    return best_time


@njit(cache=True,fastmath=False)
def refine_brent(state,wa,wb,ax,ay,witness,horizon):
    """Parabolic/golden local cut proposal; NEVER a safety acceptance.

    A bounded Brent step often resolves a smooth contact extremum with fewer
    evaluations than golden-only subdivision. Keep the initial exact witness
    if the heuristic finds no improvement. Global unimodality is not assumed
    for correctness: every cut is rebuilt at its exact proposed time and the
    same global lower/upper and continuous safety checks remain mandatory.
    """
    if not(0.<witness<=horizon):return witness
    width=min(.05,horizon/200.)
    a,b=max(1e-12,witness-width),min(horizon,witness+width)
    x=w=v=witness
    fx=fw=fv=2.*separation(state,wa,wb,x,ax,ay)/(x*x)
    d=e=0.;golden=(3.-math.sqrt(5.))/2.
    for _ in range(24):
        mid=.5*(a+b);tol1=1e-10+1e-8*abs(x);tol2=2.*tol1
        if abs(x-mid)<=tol2-.5*(b-a):break
        parabolic=False
        if abs(e)>tol1:
            rr=(x-w)*(fx-fv);qq=(x-v)*(fx-fw)
            pp=(x-v)*qq-(x-w)*rr;qq=2.*(qq-rr)
            if qq>0.:pp=-pp
            qq=abs(qq);old=e;e=d
            if qq>0. and abs(pp)<abs(.5*qq*old)and pp>qq*(a-x)and pp<qq*(b-x):
                d=pp/qq;u=x+d;parabolic=True
                if u-a<tol2 or b-u<tol2:d=math.copysign(tol1,mid-x)
        if not parabolic:e=(a-x)if x>=mid else(b-x);d=golden*e
        u=x+(d if abs(d)>=tol1 else math.copysign(tol1,d))
        fu=2.*separation(state,wa,wb,u,ax,ay)/(u*u)
        if fu<=fx:
            if u>=x:a=x
            else:b=x
            v,w,x=w,x,u;fv,fw,fx=fw,fx,fu
        else:
            if u<x:a=u
            else:b=u
            if fu<=fw or w==x:v,w=w,u;fv,fw=fw,fu
            elif fu<=fv or v==x or v==w:v=u;fv=fu
    return x
