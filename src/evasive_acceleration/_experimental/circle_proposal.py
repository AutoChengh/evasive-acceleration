"""Exact finite-polygon circle gaps, used ONLY as upper proposals."""
import math
import numpy as np
from .continuous import njit


@njit(cache=True,fastmath=False)
def _remove(lo,hi,count,a,b):
    original=count;i=0
    while i<original:
        left,right=lo[i],hi[i]
        if b<=left or a>=right:i+=1;continue
        if a<=left and b>=right:
            original-=1;count-=1;lo[i],hi[i]=lo[original],hi[original]
            if count>original:lo[original],hi[original]=lo[count],hi[count]
            continue
        if a>left:
            hi[i]=a
            if b<right:lo[count],hi[count]=b,right;count+=1
        else:lo[i]=b
        i+=1
    return count


@njit(cache=True,fastmath=False)
def propose(radius,normals,offsets,counts,npolys):
    """Largest angular gap in the union of finite collision polygons.

    Plane/circle intersections are analytic. No angular grid can miss a narrow
    finite free interval. A candidate is NOT a continuous safety certificate
    and absence of a gap is NOT a global lower-bound proof.
    """
    if radius<=0.:return 0.,0.,False
    tau=2.*math.pi
    all_lo=np.empty(34*npolys+1);all_hi=np.empty(34*npolys+1);total=0
    lo=np.empty(66);hi=np.empty(66)
    for k in range(npolys):
        count=1;lo[0],hi[0]=0.,tau
        for j in range(counts[k]):
            nx,ny=normals[k,j,0],normals[k,j,1]
            length=math.hypot(nx,ny)
            if length==0.:continue
            ratio=offsets[k,j]/(radius*length)
            if ratio>=1.:continue
            if ratio<=-1.:count=0;break
            angle=math.atan2(ny,nx)%tau;half=math.acos(ratio)
            a,b=angle-half,angle+half
            if a<0.:
                count=_remove(lo,hi,count,0.,b)
                count=_remove(lo,hi,count,a+tau,tau)
            elif b>tau:
                count=_remove(lo,hi,count,a,tau)
                count=_remove(lo,hi,count,0.,b-tau)
            else:count=_remove(lo,hi,count,a,b)
            if count==0:break
        for j in range(count):
            all_lo[total],all_hi[total]=lo[j],hi[j];total+=1
    order=np.argsort(all_lo[:total]);reach=0.;width=0.;center=0.
    for j in range(total):
        k=order[j]
        if all_lo[k]>reach:
            if all_lo[k]-reach>width:width=all_lo[k]-reach;center=.5*(all_lo[k]+reach)
        reach=max(reach,all_hi[k])
    if tau-reach>width:width=tau-reach;center=.5*(tau+reach)
    if width<=1e-13:return 0.,0.,False
    return radius*math.cos(center),radius*math.sin(center),True


@njit(cache=True,fastmath=False)
def inside_polygon(radius,vertices,count):
    """An analytic circle point INSIDE one convex CCW cell, upper-only."""
    if radius<=0.:return 0.,0.,False
    tau=2.*math.pi;lo=np.empty(2*count+4);hi=np.empty(2*count+4)
    size=1;lo[0],hi[0]=0.,tau
    for j in range(count):
        k=(j+1)%count;dx,dy=vertices[k,0]-vertices[j,0],vertices[k,1]-vertices[j,1]
        length=math.hypot(dx,dy)
        if length<=1e-14:continue
        nx,ny=dy/length,-dx/length;b=nx*vertices[j,0]+ny*vertices[j,1]
        ratio=b/radius
        if ratio>=1.:continue
        if ratio<=-1.:return 0.,0.,False
        angle=math.atan2(ny,nx)%tau;half=math.acos(ratio);a,b=angle-half,angle+half
        if a<0.:
            size=_remove(lo,hi,size,0.,b);size=_remove(lo,hi,size,a+tau,tau)
        elif b>tau:
            size=_remove(lo,hi,size,a,tau);size=_remove(lo,hi,size,0.,b-tau)
        else:size=_remove(lo,hi,size,a,b)
        if size==0:return 0.,0.,False
    at=0
    for j in range(1,size):
        if hi[j]-lo[j]>hi[at]-lo[at]:at=j
    angle=.5*(lo[at]+hi[at])
    return radius*math.cos(angle),radius*math.sin(angle),True
